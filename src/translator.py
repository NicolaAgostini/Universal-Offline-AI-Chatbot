from deep_translator import GoogleTranslator

MAX_CHARS = 4500  # google limit 5000 characters


def split_text(text: str, max_chars: int = MAX_CHARS):
    """Split testo in chunk for Google Translate"""
    chunks = []
    current = ""

    for paragraph in text.split("\n"):
        if len(current) + len(paragraph) < max_chars:
            current += paragraph + "\n"
        else:
            chunks.append(current.strip())
            current = paragraph + "\n"

    #last paragraph
    if current.strip():
        chunks.append(current.strip())

    return chunks


def translate_to_english(text: str) -> str:
    if not text or not text.strip():
        return text

    translator = GoogleTranslator(source="auto", target="en")
    chunks = split_text(text)

    translated_chunks = []
    for chunk in chunks:
        try:
            translated_chunks.append(translator.translate(chunk))
        except Exception as e:
            print("⚠️ Translation chunk failed:", e)
            translated_chunks.append(chunk)  # fallback: originale

    return "\n".join(translated_chunks)

def translate_to_italian(text: str) -> str:
    return GoogleTranslator(source='en', target='it').translate(text)


