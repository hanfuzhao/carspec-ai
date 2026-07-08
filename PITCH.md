# CarSpec AI - Pitch Script

Total: ~5 minutes (normal-slow pace, ~640 words spoken + ~75s live demo)
Aligns with PITCH_SLIDES.pptx (11 slides)

Timing breakdown:
- Slide 1 (Title): 15s
- Slide 2 (Problem): 35s
- Slide 3 (Hypothesis): 25s
- Slide 4 (Method): 40s
- Slide 5 (Pipeline): 25s
- Slide 6 (Live Demo): 75s (mostly demo action, minimal narration)
- Slide 7 (Results): 35s
- Slide 8 (Insights): 30s
- Slide 9 (Engineering): 15s
- Slide 10 (Future): 15s
- Slide 11 (Thank you): 10s

## [SLIDE 1 - Title] (15s)

Hi, I'm Hanfu. This is CarSpec AI - a system that reads a car like a spec sheet. One photo in, three attributes out, and a human-readable explanation for every call.

## [SLIDE 2 - 01 / Problem] (35s)

Vehicle attribute recognition sits inside used-car pricing, fleet inventory, insurance checks. Traditional systems do one attribute at a time and behave like a black box - you get a label, no clue how it was reached.

CarSpec AI takes both on. One model predicts type, doors, and seats together. And every prediction comes with the visual features that drove the call.

## [SLIDE 3 - 02 / Hypothesis] (25s)

The hypothesis is simple. Vehicle attributes are correlated - coupes usually have two doors and two seats, MPVs usually have five doors and seven seats. A multi-task model with a shared backbone should pick up these correlations and push accuracy up across all three tasks.

## [SLIDE 4 - 03 / Method - Three Models] (40s)

Three models, same data. Naive baseline - majority class, sanity floor. Classical - Random Forest over fifty handcrafted features: HSV histogram, HOG, LBP texture, aspect ratio, symmetry. Deep - MobileNetV2 backbone with three linear heads for car type, door count, seat count. Multi-task loss, one forward pass, three outputs.

The classical and deep models share the same fifty features, so the classical path is interpretable by construction.

## [SLIDE 5 - 04 / Pipeline] (25s)

The pipeline is four stages. Photo comes in, gets resized to 224 by 224. Features get extracted in parallel - fifty handcrafted dims go to Random Forest, the image goes to MobileNetV2. Both run, both return predictions, and the feature values get turned into plain-language sentences.

## [SLIDE 6 - 05 / Live Demo] (75s - mostly demo action)

*Switch to browser: https://hanfuzhao781-carspec-ai.hf.space*

Let me show you. I'll grab a sample from the sidebar - this sedan. Click "Run Recognition."

*Wait for result, ~3-4 seconds*

Top result: sedan, four doors, five seats - all three correct. Confidence 98 percent. On the right, the deep model and classical model agree, which is a good sign. Below that, the feature breakdown - dominant color, aspect ratio, body proportions - all point to sedan. That's the audit trail.

Let me try one more - the MPV.

*Click MPV sample, run, wait*

Type: MPV, five doors, seven seats - all three correct, confidence 97 percent. The model nails it because MPVs have a distinct tall-box silhouette that's easy to separate from sedans and SUVs.

## [SLIDE 7 - 06 / Results] (35s)

Numbers on 969 held-out real photos. Naive baseline: 23 percent on car type, 58 percent on doors and seats - that's the floor. Classical doubles car type to 41 percent. Deep wins on all three: 77 percent on car type, 82 percent on doors, 86 percent on seats. Top-5 accuracy is 100 percent, but with only three to five classes per task, that number is mostly a sanity check. Scaling from 100 to 4,869 training images pushed car_type accuracy from 0.55 to 0.77 - a 22-point gain.

## [SLIDE 8 - 07 / Insights] (30s)

Four takeaways. Multi-task lifts door and seat accuracy - the shared backbone learns car type first, then passes it through. Deep beats classical by 36 points on car type - learned features beat handcrafted on real photos. Classical doubles naive - so the fifty handcrafted features carry real signal, not noise. And under gaussian noise, accuracy drops from 0.77 to 0.35 - the classical model barely moves, which makes it a good fallback for noisy inputs.

## [SLIDE 9 - 08 / Engineering] (15s)

On the engineering side: twelve modular scripts covering crawl, clean, train, eval, and deploy, models hosted on HuggingFace Hub, Dockerized deployment, keep-alive cron, git workflow with PRs. The whole thing rebuilds from scratch in about three minutes.

## [SLIDE 10 - 09 / Future] (15s)

If I had another semester: train on the full 136k CompCars images with a GPU, add Grad-CAM heatmaps, try attention modules, fuse multiple views, and export to TensorRT for real-time inference.

## [SLIDE 11 - Thank you] (10s)

That's CarSpec AI. The app is live at the URL below, code is on GitHub. Questions?

## Demo Checklist (run before pitch)

1. Open https://hanfuzhao781-carspec-ai.hf.space in browser
2. Verify `/health` returns `models_loaded: 7`
3. Verify 5 sample images appear in sidebar
4. Pre-load the sedan sample result (run once before presenting, so the first demo call is fast)
5. Have the SUV sample ready as backup
6. Keep browser tab on the demo slide, switch with Alt-Tab / Cmd-Tab

## Fallback if demo fails

If the live site is down or slow, fall back to describing the results table on slide 7 and the feature breakdown on slide 8. The numbers and explanations are the same.
