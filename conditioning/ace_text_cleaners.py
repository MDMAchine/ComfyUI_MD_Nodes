import re
import pykakasi

# Regular expression to match commas and spaces
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for basic text cleaning
_cleaner_regexes = [
    (re.compile(r'\[(EN|JP|ZH|KO)\]'), ''),  # Remove language tags
    (re.compile(r'\[START\]'), ''),
    (re.compile(r'\[END\]'), ''),
    (re.compile(r'<break>'), ''),
    (re.compile(r'([.,!?])'), r' \1 '),  # Add space around punctuation
]

def japanese_to_romaji(text):
    """Converts Japanese text to Romaji using pykakasi."""
    kks = pykakasi.kakasi()
    # The result is a list of dictionaries, so we join the 'hepburn' values
    result = kks.convert(text)
    return ' '.join([item['hepburn'] for item in result])

def english_cleaners(text):
    """Basic cleaner for English text."""
    text = text.lower()
    for regex, replacement in _cleaner_regexes:
        text = re.sub(regex, replacement, text)
    text = re.sub(_whitespace_re, ' ', text)
    return text.strip()

def multilingual_cleaners(text, lang):
    """
    Main function to dispatch cleaning based on language.
    Currently, it applies a basic English cleaner to all inputs.
    """
    # This can be expanded with language-specific cleaners if needed
    return english_cleaners(text)