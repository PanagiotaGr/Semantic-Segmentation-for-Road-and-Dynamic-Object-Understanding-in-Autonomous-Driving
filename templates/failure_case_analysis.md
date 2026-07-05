# Failure Case Analysis Template

Use this template to analyze model errors in a structured way.

## Sample ID

- image/video id:
- dataset:
- split:
- model:
- experiment id:

## Scene Conditions

- day/night:
- weather:
- road type:
- traffic density:
- occlusion level:
- camera quality:

## Ground-Truth Classes Present

- background:
- road:
- sidewalk:
- vehicle:
- pedestrian:

## Main Failure Type

Select all that apply:

- missed pedestrian
- missed vehicle
- road/sidewalk confusion
- background confusion
- boundary error
- small-object failure
- occlusion failure
- low-light failure
- weather-related failure
- overconfident wrong prediction
- temporally unstable prediction

## Quantitative Error

| Class | IoU | False Positives | False Negatives |
| --- | ---: | ---: | ---: |
| Background | | | |
| Road | | | |
| Sidewalk | | | |
| Vehicle | | | |
| Pedestrian | | | |

## Visual Evidence

Add paths to:

- input image,
- ground-truth mask,
- predicted mask,
- overlay,
- uncertainty map,
- error map.

## Interpretation

What likely caused the failure?

## Safety Relevance

Explain whether this error could affect autonomous-driving safety.

## Proposed Fix

Possible interventions:

- higher resolution,
- class-balanced sampling,
- improved loss function,
- temporal context,
- uncertainty thresholding,
- additional data,
- domain adaptation,
- multi-task supervision.

## Follow-Up Experiment

What experiment should be run to test the proposed fix?
