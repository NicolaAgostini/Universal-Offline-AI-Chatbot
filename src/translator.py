from deep_translator import GoogleTranslator

def translate_to_english(text: str) -> str:
    return GoogleTranslator(source='auto', target='en').translate(text)

def translate_to_italian(text: str) -> str:
    return GoogleTranslator(source='en', target='it').translate(text)


