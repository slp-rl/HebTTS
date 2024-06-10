from datetime import datetime
from num2words import num2words


class HebrewNormalizer:
    HEBREW_CHARS = "אבגדהוזחטיכלמנסעפצקרשת"

    SUF_REPLACE = {
        'ף': 'פ',
        'ץ': 'צ',
        'ך': 'כ',
        'ן': 'נ',
        'ם': 'מ',
    }

    date_formats = [
        "%d.%m.%Y",
        "%d/%m/%Y",

    ]

    def __init__(self, unk_token='~', punctuation=".!?,'\"\'" + '״' + '׳'):
        self.unk_token = unk_token
        self.allowed_chars = HebrewNormalizer.HEBREW_CHARS + punctuation

    def __call__(self, text):
        normalized_text = whitespace_split(text)
        normalized_text = self.normalize(normalized_text)
        return normalized_text

    def normalize(self, text):
        res = list()
        for word in text:
            date = HebrewNormalizer.get_date(word)

            # switch to classify word
            if date:
                res += HebrewNormalizer.date_to_hebrew(date).split(" ")
            elif word.isdigit():
                res += num2words(int(word), lang='he').split(" ")
            elif self.unk_token is not None:
                res.append(self.remove_unknown_chars(word))
            else:
                res.append(word)

        return res

    def remove_unknown_chars(self, word):
        """
            remove unknown chars and "suf" characters
        """
        norm_word = ""
        for char in word:
            if char in HebrewNormalizer.SUF_REPLACE.keys():
                norm_word += HebrewNormalizer.SUF_REPLACE[char]
            elif char not in self.allowed_chars:
                norm_word += self.unk_token
            else:
                norm_word += char

        return norm_word

    @staticmethod
    def get_date(date_string):
        for date_format in HebrewNormalizer.date_formats:
            try:
                parsed_date = datetime.strptime(date_string, date_format)
                return parsed_date
            except ValueError:
                pass
        return False

    @staticmethod
    def date_to_hebrew(date: datetime):
        """
            converts dates to hebrew using num2words
            currently the gender is wrong
        """
        month = [
            "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"
        ]

        return f"{num2words(date.day, lang='he')} ב{month[date.month - 1]}, {num2words(date.year, lang='he')}"


def whitespace_split(text):
    """
        strip and split
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


if __name__ == '__main__':
    norm = HebrewNormalizer()
    texts = [
        "היום הולדת שלי ב 21.12.2002",
        "יש לי 4 בננות ו 45 ענבים",
        "עברית english"
    ]

    for text in texts:
        print(norm(text))
