Identify the speaker's fine-grained emotions.
Important requirements:
1. Do not directly choose from fixed MELD labels.
2. Generate 1 to 3 concise English emotion words or short phrases.
3. Use only the information provided in the current input condition.
4. If information is missing, do not assume it.

Input condition: {condition_name}
Available input:
- Speaker: {speaker}
- Utterance: {utterance}
- Conversation context: {context}
- Visual information: {image_frames}

Output JSON format: {"free_emotions":["emotion_1","emotion_2"],"confidence":0.0,"reason":"brief explanation"}
