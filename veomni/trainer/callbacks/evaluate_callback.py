# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dis
import json
import os
import random
import re
from collections import defaultdict
from typing import TYPE_CHECKING
import unicodedata
 
import jiwer
import opencc
from jiwer import transforms as tr 
import torch
import torch.distributed as dist
from transformers.cache_utils import DynamicCache
 
from veomni.utils.logging import get_logger
from .base import Callback, TrainerState
 
if TYPE_CHECKING:
    from ..base import VeOmniArguments
 
logger = get_logger(__name__)

ASR_CATEGORIES = frozenset({"aishell", "librispeech", "cantonese", "commonvoice_ja"})
 

class EvaluateCallback(Callback):
    """Runs evaluation at configurable step / epoch intervals.
 
    Supports two mutually-exclusive evaluation modes per run:
      • MCQ  – single forward pass, argmax over option logits.
      • ASR  – autoregressive generation, scored with WER / CER.
 
    Results are:
      1. Written to  <output_dir>/eval_results.jsonl  (rank-0 only).
      2. Printed to the console logger                (rank-0 only).
      3. Forwarded to WandbTraceCallback.on_eval_end  (if wandb is enabled).
      4. Stored in  self.trainer.eval_metrics         (all ranks, latest run).
    """
    def on_epoch_end(self, state: TrainerState, **kwargs):
        args: "VeOmniArguments" = self.trainer.args
        # if args.train.eval_epochs and (state.epoch + 1) % args.train.eval_epochs == 0:
        self._evaluate(state)

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.eval_steps and (state.global_step % args.train.eval_steps==0 or (state.global_step==1 and args.train.eval_first)):
            self._evaluate(state)
    
    @property
    def _rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0
 
    @property
    def _local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))
 
    @property
    def _device(self) -> torch.device:
        return torch.device(f"cuda:{self._local_rank}")

    def _build_option_maps(self):
        """Return (id_dict1, indices1, id_dict2, indices2) for A-J option tokens."""
        tok = self.trainer.tokenizer
        all_opts = "ABCDEFGHIJ"
        # bare letter: "A", "B", …
        id_dict1  = {tok(opt)["input_ids"][-1]: i for i, opt in enumerate(all_opts)}
        indices1  = [tok(opt)["input_ids"][-1] for opt in all_opts]
        # parenthesised: "(A", "(B", …
        id_dict2  = {tok(f"({opt}")["input_ids"][-1]: i for i, opt in enumerate(all_opts)}
        indices2  = [tok(f"({opt}")["input_ids"][-1] for opt in all_opts]
        return id_dict1, indices1, id_dict2, indices2

    def _all_reduce_mean(self, values: list, dtype=torch.float) -> float | None:
        """Local mean → all-reduce sum → divide by world_size."""
        if not values:
            return None
        local = torch.tensor(
            [v.item() if isinstance(v, torch.Tensor) else v for v in values],
            dtype=dtype, device=self._device,
        ).mean()
        if dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
            local = local / dist.get_world_size()
        return local.item()
    
    def _all_reduce_category(self, category_acc: dict, all_categories: set) -> dict:
        """Per-category right_count / total_count aggregated across all ranks."""
        results = {}
        device  = self._device
        for c in sorted(all_categories):
            acc_list = category_acc.get(c, []) 
            right = (
                torch.tensor(
                    [a.item() if isinstance(a, torch.Tensor) else float(a)
                     for a in acc_list],
                    dtype=torch.float, device=device,
                ).sum()
                if acc_list
                else torch.zeros(1, device=device).squeeze()
            )
            total = torch.tensor(len(acc_list), dtype=torch.float, device=device)
            if dist.is_initialized():
                dist.all_reduce(right, op=dist.ReduceOp.SUM)
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
            results[c] = (right / total).item() if total.item() > 0 else 0.0
        return results
    

    def _eval_mcq_batch(
        self, model, data: dict,
        id_dict1, indices1, id_dict2, indices2,
    ):
        """Single forward, argmax over option logits.
 
        Returns (acc, prob, all_acc, all_prob) tensors, or None on error.
        All ranks call model() exactly once → safe with FSDP2.
        """
        options_num = data.pop("options_num")

        rank = self._rank
    
        output = model(**data)
       
        raw_target = data["labels"][0, -1].item()
        if raw_target in id_dict1:
            target_idx    = id_dict1[raw_target]
            option_indices = indices1
        elif raw_target in id_dict2:
            target_idx    = id_dict2[raw_target]
            option_indices = indices2
        else:
            logger.warning(
                f"[Eval MCQ] Unrecognized target token id={raw_target}, "
                "falling back to random choice."
            )
            target_idx    = random.randint(0, max(options_num - 1, 0))
            option_indices = indices1
 
        try:
            opt_logits = output["logits"][0, -2, option_indices[:options_num]]
            opt_probs  = torch.softmax(opt_logits, dim=-1)
            pred_idx = opt_logits.argmax().item()  # 获取模型选择的索引
            acc  = opt_logits.argmax() == target_idx
            prob = opt_probs[target_idx]
 
            # all-vocab metrics (logit rank over the entire vocabulary)
            all_logits = output["logits"][0, -2]
            all_probs  = torch.softmax(all_logits, dim=-1)
            all_acc    = all_logits.argmax() == raw_target
            all_prob   = all_probs[raw_target]

           
            return acc, prob, all_acc, all_prob, pred_idx, target_idx
 
        except Exception as exc:
            logger.warning(f"[Eval MCQ] Batch failed: {exc}")
            return None
 
    def _eval_asr_batch(self, model, data: dict) -> tuple[float, torch.device]:
        """Pop ASR-specific keys from data and run generation-based scoring.
 
        All ranks execute exactly MAX_GEN forwards → safe with FSDP2,
        provided the entire eval set consists only of ASR samples.
        Returns (metric, device).
        """
        input_ids      = data.pop("input_ids")
        gt_str, lang   = data.pop("audio_ground_truth_text")[0]
        audio          = data.pop("audio_features")
        audio_feat_len = data.pop("audio_features_lens")
 
        return self._generate_and_score(
            model, input_ids, gt_str, lang, audio, audio_feat_len,
        )
    
    def _generate_and_score(
        self, model, input_ids, gt_str: str, language: str,
        audios, audio_feature_len,
    ) -> tuple[float, torch.device]:
        """Greedy / top-p/k generation loop followed by WER / CER scoring."""
        
        tokenizer  = self.trainer.tokenizer
        device     = self._device
        MAX_GEN    = 32
        REP_PEN    = 1.05
        TEMP       = 0.0
        TOP_K      = 20
        TOP_P      = 0.8
        FILTER_VAL = -float("Inf")
 
        cc = opencc.OpenCC("s2hk")
 
        # ── text-normalisation helpers ────────────────────────────────────────
        def _lang_char_transform(text: str, lang: str) -> list[str]:
            text = unicodedata.normalize("NFKC", text)
            if lang == "zh_yue":
                text = cc.convert(text).lower()
            patt      = re.compile(r"[a-zA-Z0-9]+|[^a-zA-Z0-9\s\u0000-\u007f]")
            punct_re  = re.compile(r"[^\w]", re.UNICODE)
            return [
                w for w in patt.findall(text)
                if not (len(w) == 1 and punct_re.match(w) and not w.isalnum())
            ]
 
        def _mixed_transform(texts: list[str]) -> list[list[str]]:
            return [
                re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", s)
                for s in texts
            ]
 
        _wer_transforms = tr.Compose([
            tr.ToLowerCase(),
            tr.RemoveMultipleSpaces(),
            tr.RemovePunctuation(),
            tr.Strip(),
            tr.ExpandCommonEnglishContractions(),
            tr.ReduceToListOfListOfWords(),
        ])
 
        def _score(gt: str, hyp: str, lang: str) -> float:
            if lang == "zh":
                return jiwer.cer(
                    gt, hyp,
                    reference_transform=_mixed_transform,
                    hypothesis_transform=_mixed_transform,
                )
            elif lang == "en":
                return jiwer.wer(
                    gt, hyp,
                    reference_transform=_wer_transforms,
                    hypothesis_transform=_wer_transforms,
                )
            elif lang in ("zh_yue", "jap"):
                gt_t = _lang_char_transform(gt, lang)
                hp_t = _lang_char_transform(hyp, lang)
                return jiwer.process_words(
                    reference=[gt_t], hypothesis=[hp_t],
                    reference_transform=tr.Compose([]),
                    hypothesis_transform=tr.Compose([]),
                ).wer
            else:
                raise NotImplementedError(f"[Eval ASR] language '{lang}' not supported")
 
        # ── generation loop ───────────────────────────────────────────────────
        # KV-cache is used so only the new token is passed after the first step.
        # audio features are passed only on the first step (they are embedded into
        # the KV cache via the audio encoder); subsequent steps skip them.
        cur_ids    = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
        past_kv    = DynamicCache()
        repeat_ids = torch.empty(0, dtype=torch.long, device=device)
        output_ids: list[int] = []
        gen_device = None
 
        for step in range(MAX_GEN):
            model_kwargs = dict(
                input_ids=cur_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            # Audio inputs are only meaningful on the first decode step.
            if step == 0:
                model_kwargs["audio_features"] = audios
                model_kwargs["audio_features_lens"] = audio_feature_len
 
            out     = model(**model_kwargs)
            past_kv = out.past_key_values
            logit   = out.logits[0, -1]
 
            if gen_device is None:
                gen_device = logit.device
            repeat_ids = repeat_ids.to(logit.device)
 
            # repetition penalty
            scores = torch.gather(logit, 0, repeat_ids)
            scores = torch.where(scores < 0, scores * REP_PEN, scores / REP_PEN)
            logit  = torch.scatter(logit, 0, repeat_ids, scores)
 
            # top-k filtering
            k         = min(TOP_K, logit.size(0))
            threshold = torch.topk(logit, k)[0][-1]
            logit     = logit.masked_fill(logit < threshold, FILTER_VAL)
 
            # top-p (nucleus) filtering
            sorted_l, sorted_i = torch.sort(logit, descending=False)
            cum_p     = sorted_l.softmax(-1).cumsum(-1)
            to_remove = cum_p <= (1 - TOP_P)
            to_remove[-1:] = False   # always keep at least one token
            logit = logit.masked_fill(
                to_remove.scatter(0, sorted_i, to_remove), FILTER_VAL,
            )
 
            # sample / greedy
            if TEMP == 0.0:
                tok = logit.argmax().item()
            else:
                tok = torch.multinomial(
                    torch.nn.functional.softmax(logit / TEMP, dim=-1),
                    num_samples=1,
                ).item()
 
            new_tok    = torch.tensor([[tok]], dtype=torch.long, device=gen_device)
            cur_ids    = new_tok                               # only the new token next iter
            repeat_ids = torch.cat([repeat_ids, new_tok[0]])
            output_ids.append(tok)
 
        # trim at the first EOS / BOS / PAD token
        special_ids = {
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.pad_token_id,
        } - {None}
        for i, tok in enumerate(output_ids):
            if tok in special_ids:
                output_ids = output_ids[:i]
                break
 
        hypothesis = tokenizer.decode(output_ids, skip_special_tokens=True)
        metric     = _score(gt_str, hypothesis, language)
        return metric, gen_device or device, hypothesis

    def _evaluate(self, state: TrainerState) -> None:
        args   = self.trainer.args
        model  = self.trainer.model
        rank   = self._rank
        device = self._device
 
        eval_dl = getattr(self.trainer, "eval_dataloader", None)
        if eval_dl is None:
            logger.warning_rank0("[Eval] No eval dataloader (trainer.eva_dataloader) found, skipping.")
            return
 
        logger.info_rank0(
            f"[Eval] Starting evaluation at step={state.global_step} epoch={state.epoch:.4f}"
        )
 
        id_dict1, indices1, id_dict2, indices2 = self._build_option_maps()
 
        prob_results:      list = []
        acc_results:       list = []
        all_prob_results:  list = []
        all_acc_results:   list = []
        category_acc: dict      = defaultdict(list)
        local_predictions: list = []
 
        model.eval()

        with torch.no_grad():
            for idx, data in enumerate(eval_dl):
                if rank==0 and idx % 2 == 0:
                    logger.info(f"[Eval] rank: {rank} step={state.global_step}  idx={idx}")
                
                raw_data = data.pop("raw_data", None)[0]
                # ── device transfer ───────────────────────────────────────
                bf16_keys = {"images", "pixel_values", "pixel_values_videos", "audio_features"}
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        tgt_dtype = torch.bfloat16 if k in bf16_keys else v.dtype
                        data[k]   = v.to(device=device, dtype=tgt_dtype)
 
                category: str = data.pop("category")[0]
                # Skip dummy batches produced by the collator when all samples
                # in a batch are None (e.g. failed to load).  We still need to
                # run model() so that FSDP all-gathers stay in sync across ranks.
                if category == "dummy_skip":
                    output = model(**data)
                    continue
                pred_record = raw_data.copy()
                # ── dispatch to eval mode ─────────────────────────────────
                if category in ASR_CATEGORIES:
                    # generation-based: MAX_GEN forwards per sample
                    metric, _, hypothesis= self._eval_asr_batch(model, data)
                    category_acc[category].append(
                        torch.tensor(metric, dtype=torch.float, device=device)
                    )
                    pred_record["prediction"] = hypothesis
                    pred_record["metric"] = metric
 
                else:
                    # MCQ: 1 forward per sample
                    result = self._eval_mcq_batch(
                        model, data, id_dict1, indices1, id_dict2, indices2,
                    )
                    if result is not None:
                        acc, prob, all_acc, all_prob, pred_idx, target_idx= result
                        acc_results.append(acc)
                        prob_results.append(prob)
                        all_acc_results.append(all_acc)
                        all_prob_results.append(all_prob)
                        category_acc[category].append(acc)
                        if "wukong" in category:
                            category_acc["wukong"].append(acc)
 
                        pred_record["prediction"] = chr(ord('A') + pred_idx)
                        pred_record["ground_truth"] = chr(ord('A') + target_idx)
                        pred_record["is_correct"] = acc.item() if isinstance(acc, torch.Tensor) else acc
                
                if getattr(self.trainer.args, "save_predictions", True):
                    local_predictions.append(pred_record)

        if dist.is_initialized():
            dist.barrier()

        if getattr(self.trainer.args, "save_predictions", True):
            if dist.is_initialized():
                gather_list = [None] * dist.get_world_size()
                dist.all_gather_object(gather_list, local_predictions)
                if rank == 0:
                    all_predictions = [item for sublist in gather_list for item in sublist]
            else:
                all_predictions = local_predictions
 
            if rank == 0:
                out_dir = args.train.checkpoint.output_dir
                os.makedirs(out_dir, exist_ok=True)
                pred_file = os.path.join(out_dir, f"eval_predictions_step{state.global_step}.json")
                with open(pred_file, "w", encoding="utf-8") as f:
                    json.dump(all_predictions, f, ensure_ascii=False, indent=4)
                logger.info(f"[Eval] Saved consolidated predictions to {pred_file}")
 
        # ── aggregate across ranks ────────────────────────────────────────
        evaluate_logs: dict = {}
 
        for metric_key, values in [
            ("prob",     prob_results),
            ("acc",      acc_results),
            ("all_prob", all_prob_results),
            ("all_acc",  all_acc_results),
        ]:
            val = self._all_reduce_mean(values)
            if val is not None:
                evaluate_logs[metric_key] = val

        all_categories = getattr(eval_dl, "categories", set())
        evaluate_logs.update(self._all_reduce_category(category_acc, all_categories))
        evaluate_logs["step"]  = state.global_step
        evaluate_logs["epoch"] = round(float(state.epoch), 4)
 
        # ── log & persist ─────────────────────────────────────────────────
        self._log_eval_results(evaluate_logs, state)
 
        # Expose for trace callbacks (WandbTraceCallback, EnvironMeterCallback).
        self.trainer.eval_metrics = evaluate_logs
 
        model.train()

    
    def _log_eval_results(self, evaluate_logs: dict, state: TrainerState) -> None:
        """Write results to JSONL, print to console, and push to wandb."""
        args = self.trainer.args
        rank = self._rank
 
        if rank == 0:
            out_dir = args.train.checkpoint.output_dir
            os.makedirs(out_dir, exist_ok=True)
            jsonl_path = os.path.join(out_dir, "eval_results.jsonl")
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(evaluate_logs, ensure_ascii=False) + "\n")
 
        if rank == 0:
            core_metrics = {
                k: v for k, v in evaluate_logs.items() if k not in ("step", "epoch")
            }
            summary = "  ".join(f"{k}: {v:.4f}" for k, v in core_metrics.items())
            logger.info(
                f"[Eval] step={evaluate_logs['step']} | "
                f"{summary}"
            )
 
        # 3. Wandb — delegate to WandbTraceCallback so all wandb logic stays there
        wandb_cb = getattr(self.trainer, "wandb_callback", None)
        if wandb_cb is not None:
            wandb_cb.on_eval_end(state, eval_metrics=evaluate_logs)
        tensorboard_cb = getattr(self.trainer, "tensorboard_callback", None)
        if tensorboard_cb is not None:
            tensorboard_cb.on_eval_end(state, eval_metrics=evaluate_logs)
