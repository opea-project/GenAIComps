import os

def detect_phones(text):
    """Detects phone in a string using phonenumbers libray only detection the international phone number"""
    try:
        import phonenumbers
    except ImportError:
        os.system("pip install phonenumbers")
        import phonenumbers
        
    matches = []

    for match in phonenumbers.PhoneNumberMatcher(text, "IN"):
        matches.append(
            {
                "tag": "PHONE_NUMBER",
                "value": match.raw_string,
                "start": match.start,
                "end": match.end,
            }
        )
    return matches
