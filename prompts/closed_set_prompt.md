Classify the speaker's emotion into exactly one MELD label.
Allowed labels: anger, disgust, sadness, joy, neutral, surprise, fear.
Use the utterance as primary evidence and use context or visual information when available.

Input condition: text + context + image
Available input:
- Speaker: {speaker}
- Utterance: {utterance}
- Conversation context: {context}
- Visual information: {image_frames}

Output JSON format: {"predicted_label":"one allowed label","confidence":0.0,"reason":"brief explanation"}
