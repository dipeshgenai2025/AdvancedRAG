# Utils/utils.py
import re
import logging

def format_text_by_sentences(text: str, max_words: int = 100) -> str:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if not isinstance(max_words, int) or max_words <= 0:
        raise ValueError("max_words must be a positive integer")

    try:
        # Remove line breaks
        text = " ".join(line.strip() for line in text.splitlines() if line.strip())

        combined_sentences = []
        buffer = ""
        sentence = ""
        paren_level = 0  # Tracks nesting in () or []

        index = 0
        while index < len(text):
            char = text[index]
            sentence += char

            # Track parentheses and brackets
            if char in "([":
                paren_level += 1
            elif char in ")]":
                paren_level = max(paren_level - 1, 0)

            # Check if this is a sentence boundary
            if char in ".;" and paren_level == 0:
                lookahead = text[index+1:index+3] if index+3 <= len(text) else text[index+1:]
                if re.match(r'\s+[A-Z0-9]', lookahead) or index == len(text)-1:
                    if len((buffer + " " + sentence).split()) > max_words:
                        if buffer:
                            combined_sentences.append(buffer.strip())
                        buffer = sentence
                    else:
                        buffer = (buffer + " " + sentence).strip()
                    sentence = ""

            index += 1

        # Append any remaining text
        if sentence:
            if len((buffer + " " + sentence).split()) > max_words:
                if buffer:
                    combined_sentences.append(buffer.strip())
                combined_sentences.append(sentence.strip())
            else:
                buffer = (buffer + " " + sentence).strip()

        if buffer:
            combined_sentences.append(buffer.strip())

        # Ensure picture descriptions are separated
        final_sentences = []
        for sent in combined_sentences:
            # Split before "Picture <number>:" so each starts fresh
            parts = re.split(r'(Picture\s+\d+\s+:)', sent)
            merged = []
            idx = 0
            while idx < len(parts):
                if re.match(r'Picture\s+\d+\s+:', parts[idx]):
                    if idx + 1 < len(parts):
                        merged.append(parts[idx] + parts[idx+1].strip())
                        idx += 2
                    else:
                        merged.append(parts[idx].strip())
                        idx += 1
                else:
                    if parts[idx].strip():
                        merged.append(parts[idx].strip())
                    idx += 1
            final_sentences.extend(merged)

        return "\n".join(final_sentences)
    except Exception as e:
        logging.error(f"Error in format_text_by_sentences: {e}")
        raise RuntimeError(f"Text formatting failed: {e}")