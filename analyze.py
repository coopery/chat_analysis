#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import numpy as np
import re
import string

from bad_words import BAD_WORDS
from datetime import datetime
from nltk.corpus import stopwords

CHAT_CSV_FILENAME = 'allo_chat_messages_2018-12-17_11_26_35_PST_edited.csv'
TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S %Z'

COOPER_SENDER = '+15712711205'
MAGGIE_SENDER = '+12489046477 (Large Marge)'
SENDER_MAP = {
    COOPER_SENDER: 'Cooper',
    MAGGIE_SENDER: 'Maggie',
}

URL_REGEX = re.compile(
    r'^(?:http|ftp)s?://' # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
    r'localhost|' #localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
    r'(?::\d+)?' # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)

STOP_WORDS = set(stopwords.words('english'))


##### functions to get data from a row

def get_sender(row):
    return row[3]

def sent_by_cooper(row):
    return get_sender(row) == COOPER_SENDER

def sent_by_maggie(row):
    return get_sender(row) == MAGGIE_SENDER

def get_message(row):
    return row[5]

def is_text(row):
    # could also be image, sticker, etc
    return row[4] == 'text'

def is_url(row):
    return re.match(URL_REGEX, get_message(row)) is not None

def is_dumb_sender(row):
    return get_sender(row) in ['Google Assistant', 'Lucky']

def get_words(row):
    words = get_message(row) \
                .translate(str.maketrans('', '', string.punctuation)) \
                .split(' ')
    return [w.lower() for w in words]


##### counting base classes

class RateTracker():
    def __init__(self):
        self.xs = []
        self.ys = []

    def extract_date(self, row):
        # parse string and drop time of day
        return datetime.strptime(row[0], TIMESTAMP_FORMAT).date()

    def extract_value(self, row):
        pass

    def record_datum(self, row):
        x = self.extract_date(row)
        y = self.extract_value(row)
        if y is None:
            return
        if len(self.xs) > 0 and self.xs[-1] == x:
            self.ys[-1] += y
        else:
            self.xs.append(x)
            self.ys.append(y)

    def print_maximum(self):
        max_i = np.argmax(self.ys)
        print('{} max: day {}, count {}'.format(type(self).__name__,self.xs[max_i], self.ys[max_i]))

    def maximum(self):
        return max(self.ys)

    def average(self, window_days):
        index = 0
        new_ys = [None] * len(self.ys)
        while index < len(self.ys):
            if index < window_days:
                new_ys[index] = self.ys[index]
            new_ys[index] = np.average(self.ys[index+1 - window_days:index+1])
            index += 1
        self.ys = new_ys

class LengthTracker(RateTracker):
    def extract_date(self, row):
        # parse string and return date and time
        return datetime.strptime(row[0], TIMESTAMP_FORMAT)

    def extract_value(self, row):
        return len(row[5])

class WordTracker():
    def __init__(self):
        # { sender: { word: count } }
        self.word_count = {}

    def record_datum(self, row):
        if is_url(row) or is_dumb_sender(row) or not is_text(row):
            return
        sender = SENDER_MAP[get_sender(row)]
        if sender not in self.word_count:
            self.word_count[sender] = {}
        for word in get_words(row):
            if word in STOP_WORDS:
                continue
            if word in self.word_count[sender]:
                self.word_count[sender][word] += 1
            else:
                self.word_count[sender][word] = 1


##### texts per day trackers

class CooperRateTracker(RateTracker):
    fmt = 'b'
    label = 'Cooper'
    def extract_value(self, row):
        return int(sent_by_cooper(row))

class MaggieRateTracker(RateTracker):
    fmt = 'g'
    label = 'Maggie'
    def extract_value(self, row):
        return int(sent_by_maggie(row))

class LoveRateTracker(RateTracker):
    fmt = 'r'
    label = 'Texts with "love you"'
    def extract_value(self, row):
        msg = get_message(row) if is_text(row) else ''
        return int('love you' in msg.lower())

class CussRateTracker(RateTracker):
    fmt = 'k'
    label = 'Texts with cuss words'
    def extract_value(self, row):
        if not is_text(row):
            return
        words = get_words(row)
        for word in words:
            if word in BAD_WORDS:
                return 1
        return 0

class CooperCussRateTracker(CussRateTracker):
    fmt = 'c'
    label = "Cooper's texts with cuss words"
    def extract_value(self, row):
        if sent_by_cooper(row):
            return super(CooperCussRateTracker, self).extract_value(row)
        return 0

class MaggieCussRateTracker(CussRateTracker):
    fmt = 'm'
    label = "Maggie's texts with cuss words"
    def extract_value(self, row):
        if sent_by_maggie(row):
            return super(MaggieCussRateTracker, self).extract_value(row)
        return 0


##### length of text trackers

class CooperLengthTracker(LengthTracker):
    label = 'Cooper'
    def extract_value(self, row):
        if sent_by_cooper(row) and not is_url(row):
            return len(get_message(row))
        return None

class MaggieLengthTracker(LengthTracker):
    label = 'Maggie'
    def extract_value(self, row):
        if sent_by_maggie(row) and not is_url(row):
            return len(get_message(row))
        return None


##### input/output

def scan_file(filename, data_trackers):
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        # ignore header row
        next(reader, None)
        index = 1

        for row in reader:
            for dt in data_trackers:
                try:
                    dt.record_datum(row)
                except Exception as e:
                    print(f'Error on row {index}, tracker: {dt}, {e}')
            index += 1

def plot_time_series(data_trackers, window):
    for data in data_trackers:
        plt.plot(data.xs, data.ys, data.fmt, label=data.label)
    plt.xlabel('Day')
    plt.ylabel('Texts')
    plt.legend()
    plt.suptitle("Mooper's Chat History", fontsize=18)
    if window is not None:
        plt.title(f'{window} day moving average', fontsize=10)
    plt.show()

def plot_histogram(data_trackers):
    bins = np.arange(1, 200)
    for data in data_trackers:
        plt.hist(data.ys, bins, alpha=0.5, label=data.label)
    plt.xlabel('Text Length')
    plt.ylabel('Number of Texts')
    plt.legend()
    plt.suptitle("Mooper's Text Lengths", fontsize=18)
    plt.show()

def print_word_list(data_trackers):
    for dt in data_trackers:
        for sender, word_count in dt.word_count.items():
            print('**********************')
            print(f'Word counts for {sender}:')
            sorted_words = [(k, word_count[k]) for k in sorted(word_count, key=word_count.get, reverse=True)]
            index = 0
            for word, count in sorted_words:
                if index > 50:
                    break
                print(f'{word}:\t{count}')
                index += 1


if __name__ == '__main__':
    data_trackers = [
            CooperRateTracker(),
            MaggieRateTracker(),
            CussRateTracker(),
            LoveRateTracker(),
        ]

    scan_file(CHAT_CSV_FILENAME, data_trackers)

    for dt in data_trackers:
        dt.print_maximum()

    window = 30
    for dt in data_trackers:
        dt.average(window)

    plot_time_series(data_trackers, window)
#    plot_histogram(data_trackers)
#    print_word_list(data_trackers)
