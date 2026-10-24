import re

# Weights for different flags (total cap 100)
WEIGHTS = {
    "urgency": 20,
    "impersonation": 15,
    "otp_payment": 30,
    "suspicious_link": 25,
    "hinglish": 10
}

# Thresholds
HINGLISH_COUNT_THRESHOLD = 2

# Keyword lists and regexes
URGENCY_KEYWORDS = ["urgent", "immediately", "within 24 hours", "blocked", "suspended", "expire", "last chance", "legal action", "arrest", "turant", "jaldi", "abhi", "band ho jayega"]
IMPERSONATION_KEYWORDS = ["sbi", "hdfc", "icici", "axis", "pnb", "paytm", "phonepe", "gpay", "upi", "rbi", "aadhaar", "uidai", "pan", "income tax", "trai", "india post", "electricity board", "kyc update", "deactivation"]
OTP_PAYMENT_KEYWORDS = ["otp", "pin", "cvv", "password", "send money", "collect request", "pay", "processing fee", "refund", "claim reward", "prize", "cashback", "lottery", "kbc"]
HINGLISH_KEYWORDS = ["aap", "aapka", "karo", "kijiye", "turant", "paisa", "khata", "inaam", "jeeta", "band", "jaldi"]

# Regex for URLs and specific suspicious links
URL_REGEX = r"(?i)\b(?:https?://|www\.)\S+\b|\b\S+\.(?:com|in|org|net|xyz|top|click|icu)\b"
SUSPICIOUS_DOMAIN_REGEX = r"(?i)(bit\.ly|tinyurl|\.xyz|\.top|\.click|\.icu|http://)"
IP_REGEX = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
DEVANAGARI_REGEX = r"[\u0900-\u097F]"
PHONE_REGEX = r"\b\d{10}\b"

def clean_text(message):
    """
    Cleans the input text by lowercasing, collapsing whitespace, and stripping extra punctuation,
    while keeping URLs and digits intact.
    """
    text = message.lower()
    text = re.sub(r'\s+', ' ', text)
    # Keep alphanumeric, basic punctuation used in URLs/amounts, and spaces
    text = re.sub(r'[^\w\s\.\:\/\-\?\=\&]', '', text)
    return text.strip()

def analyze_linguistics(message):
    """
    Analyzes the message for scam linguistic patterns based on rules.
    Returns a dictionary with flags, matched keywords, and a computed risk score.
    """
    cleaned_message = clean_text(message)

    flags = {
        "urgency": False,
        "impersonation": False,
        "otp_payment": False,
        "suspicious_link": False,
        "hinglish": False
    }
    matched = {flag: [] for flag in flags}

    # Check Urgency
    for kw in URGENCY_KEYWORDS:
        if kw in cleaned_message:
            flags["urgency"] = True
            matched["urgency"].append(kw)

    # Check Impersonation
    for kw in IMPERSONATION_KEYWORDS:
        if kw in cleaned_message:
            flags["impersonation"] = True
            matched["impersonation"].append(kw)

    # Check OTP/Payment
    for kw in OTP_PAYMENT_KEYWORDS:
        if kw in cleaned_message:
            flags["otp_payment"] = True
            matched["otp_payment"].append(kw)

    # Check Suspicious Links
    urls = re.findall(URL_REGEX, message) # Search in original message to keep case/structure for regex
    for url in urls:
        if re.search(SUSPICIOUS_DOMAIN_REGEX, url) or re.search(IP_REGEX, url):
            flags["suspicious_link"] = True
            matched["suspicious_link"].append(url)

    # If a bare 10-digit phone number is asked to be called
    phones = re.findall(PHONE_REGEX, cleaned_message)
    if phones and ("call" in cleaned_message or "whatsapp" in cleaned_message or "contact" in cleaned_message):
        flags["suspicious_link"] = True
        matched["suspicious_link"].extend(phones)

    # Check Hinglish
    hinglish_count = 0
    for kw in HINGLISH_KEYWORDS:
        # Match as whole word to avoid false positives
        if re.search(rf"\b{kw}\b", cleaned_message):
            hinglish_count += 1
            matched["hinglish"].append(kw)

    devanagari_chars = re.findall(DEVANAGARI_REGEX, message)
    if devanagari_chars:
         hinglish_count += len(devanagari_chars)
         matched["hinglish"].extend(list(set(devanagari_chars)))

    if hinglish_count >= HINGLISH_COUNT_THRESHOLD:
        flags["hinglish"] = True

    # Compute Score
    score = 0
    for flag, is_set in flags.items():
        if is_set:
            score += WEIGHTS.get(flag, 0)

    score = min(score, 100) # Cap at 100

    return {
        "flags": flags,
        "matched": matched,
        "score": score
    }
