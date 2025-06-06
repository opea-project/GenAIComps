# Dictionary mapping language codes to their full language names.
# This dictionary is used with the ftlangdetect package to identify languages based on their ISO 639-1 language codes. 
language_dict = {
    'ar': 'Arabic',
    'cs': 'Czech',
    'da': 'Danish',
    'en': 'English',
    'et': 'Estonian',
    'fi': 'Finnish',
    'fr': 'French',
    'de': 'German',
    'el': 'Greek',
    'he': 'Hebrew',
    'hu': 'Hungarian',
    'it': 'Italian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'no': 'Norwegian',
    'fa': 'Persian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sk': 'Slovak',
    'es': 'Spanish',
    'sv': 'Swedish',
    'zh': 'Chinese',
}


def get_prompt_template() -> str:
    """Returns a tuple containing prompt system_prompt_template and user_prompt_template."""
    system_prompt_template = """
            Translate this from {source_lang} to {target_lang}:
            {source_lang}:
        """
    user_prompt_template = """
            {text}

            {target_lang}:
        """

    return system_prompt_template, user_prompt_template


def get_language_name(lang_code: str) -> str:
    """Return the language name for a language code. Returns empty string for unsupported language code."""
    return language_dict.get(lang_code, "")


def validate_language_name(lang_name: str) -> bool:
    """Returns True only if provided language name is a valid language."""
    return lang_name in language_dict.values()