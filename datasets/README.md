# Evaluation indexes

The JSON/JSONL indexes registered in `scripts/eval_sglang.py` are grouped by
primary media type. Filenames use lowercase registry names.

| Directory | Indexes | Benchmarks |
| --- | ---: | --- |
| `audio/` | 12 | LibriSpeech (four splits), Wenet, AISHELL, CommonVoice 15 zh, CommonVoice 17 zh/en, test_meeting, test_net, online_badaudio |
| `image/` | 12 | MMStar, MMMU-dev, RealWorldQA, CRPE, MMhalBench, RefoMB, OCR, OCR generate, TextVQA, DocVQA, ScreenSpot, ChartQA |
| `video/` | 4 | MVBench, LVBench, VideoMME, VideoMMMU |

Registry paths are relative to the repository root and resolved from the
script location, so `--eval_dataset` works from any working directory. Custom
`--dataset_path` inputs remain relative to the working directory. `docvqa`
aliases `image/docqa.json`.

The 28 indexes contain 114,282 records in total. Records and media references
are copied unchanged. Migration does not query S3, and original source files
are retained for other scripts. The missing `SyncDoc` and `SyncDoc_complex`
indexes have been removed from the registry.

```bash
conda activate llava
python scripts/eval_sglang.py --eval_dataset textvqa --temperature 0
python scripts/eval_sglang.py --eval_dataset mvbench --temperature 0
```
