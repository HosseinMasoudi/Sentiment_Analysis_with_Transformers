import string
import re
import unicodedata
from num2fawords import words

def normalize_unicode(text):
    """Unicode Normalization"""
    return unicodedata.normalize('NFC', text)

def remove_unwanted_characters(text):
    """Remove unwanted characters, keeping only Persian"""
    text = re.sub(r'[^\x00-\x7F\u0600-\u06FF\s]', '', text)
    return text

def convert_numbers(text, to_english=True):
    """Convert Persian digits to English"""
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    english_digits = '0123456789'
    table = str.maketrans(persian_digits if to_english else english_digits,
                          english_digits if to_english else persian_digits)
    return text.translate(table)

def standardize_persian_text(text):
    """Standardize Persian text"""
    text = text.replace('ي', 'ی').replace('ك', 'ک')  
    text = text.replace('\u200C', '')
    return text

def remove_punctuation(text):
    """Remove punctuation from text"""
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~،؟»«؛٫٬'''
    return ''.join(char for char in text if char not in punctuations)

def remove_keshide(text):
    """Remove keshideh (ـ) from text"""
    return re.sub(r'(ـ)+', '', text)

# Read the stopwords file and put each word into a list
with open("/Users/hossein/Desktop/Hos/CODE/NLP/Transformes/Persian_StopList.txt", encoding="utf-8") as file:
    stopwords = [line.strip() for line in file if line.strip()]

def remove_stopwords(text):
    """Remove Persian stopwords"""
    words = text.split()
    return ' '.join([word for word in words if word not in stopwords])

def fix_persian_zwnj(text):
    """Fix ZWNJ (Zero Width Non-Joiner) issues in Persian text"""
    text = re.sub(r'\b(ن?می|خواه(?:م|ی|د|یم|ید|ند))[\s‌]+', r'\1‌', text)

    suffixes = ['تر', 'ترین', 'ها', 'های', 'ام', 'ات', 'اش', 'ای', 'اید', 'ایم', 'اند', 'ایی', 'مان', 'تان', 'شان']
    suffix_pattern = r'(\S+)[\s‌]+(' + '|'.join(suffixes) + r')\b'
    
    text = re.sub(suffix_pattern, r'\1‌\2', text)

    text = re.sub(r'(\d+)[\s‌]+([آ-ی]+)', r'\1‌\2', text)

    text = re.sub(r'([آ-ی]+)[\s‌]+(پوش|دوست|گونه|نما|فام|سان|گر|کار|مند|ساز|تر|وار)\b', r'\1‌\2', text)
    return text

def convert_numbers_to_words(text):
    """Convert numbers in text to Persian words."""
    def replacer(match):
        number = match.group()
        try:
            return words(int(number))
        except ValueError:
            return number 
        
    return re.sub(r'\d+', replacer, text)

def normalize_persian_text(text):
    text = normalize_unicode(text)
    text = remove_unwanted_characters(text)
    text = convert_numbers(text, to_english=True)
    text = convert_numbers_to_words(text)
    text = standardize_persian_text(text)
    text = remove_keshide(text)
    text = remove_punctuation(text)
    text = fix_persian_zwnj(text)
    text = remove_stopwords(text)
    return text

print("Normalizer module loaded successfully.")
