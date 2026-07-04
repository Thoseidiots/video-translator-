# Todo: Translate Japanese Video to English (Visual Replacement)

## Phase 1: Optimize pipeline (critical fixes)
- [x] Kill stuck Part 1 process
- [x] Optimize build_ffmpeg_filters: replace per-result sample_bg_color with batch approach
- [x] Add missing deduplicate_ocr_results call to main() pipeline
- [x] Add more aggressive filtering for noise (short fragments, pure numbers, etc.)
- [x] Test optimized pipeline on short clip (30s test: verified Prison School title + UI labels)

## Phase 2: Process all 4 split videos
- [x] Part 1: videocomp_xb9n6fgx6h.mp4 (title card, 270s, 5fps) — DONE, 17.5MB
- [ ] Part 2: videocomp_soadcidkvo.mp4 (drawing part 1, 271s, 5fps)
- [ ] Part 3: videocomp_zd4c7s2nag.mp4 (drawing part 2, 300s, 9fps)
- [ ] Part 4: videocomp_u22zueh7ej.mp4 (drawing part 3, 331s, 9fps)

## Phase 3: Deliver results
- [ ] Verify each translated video by extracting sample frames
- [ ] Deliver all 4 translated videos to user
