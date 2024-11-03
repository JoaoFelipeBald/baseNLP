import nltk
from nltk.corpus import wordnet
import random

# Download necessary data if not already present
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
random.seed(42)

def get_pos_tags(words):
    """Get WordNet POS tags for a list of words."""
    tagged_words = nltk.pos_tag(words)
    pos_tags = []
    for word, tag in tagged_words:
        tag = tag[0].lower()  # Get the first character to map to WordNet POS
        if tag == 'n':
            pos_tags.append((word, wordnet.NOUN))
        elif tag == 'v':
            pos_tags.append((word, wordnet.VERB))
        elif tag == 'j':  # NLTK uses 'J' for adjectives
            pos_tags.append((word, wordnet.ADJ))
        elif tag == 'r':
            pos_tags.append((word, wordnet.ADV))
        else:
            pos_tags.append((word, None))  # For unrecognized tags
    return pos_tags

def synonym_replacement(text):
    """Replace some words in the text with their synonyms."""
    words = text.split()
    new_words = words.copy()
    num_replacements = max(1, len(words) // 9)  # Replace 20% of words

    pos_tagged_words = get_pos_tags(words)  # Get POS tags for all words at once
    indices = random.sample(range(len(words)), num_replacements)

    for i in indices:
        word, pos = pos_tagged_words[i]
        if pos:  # Check if POS tag is valid
            synonyms = wordnet.synsets(word, pos=pos)
            if synonyms:
                synonym = random.choice(synonyms).lemmas()[0].name()
                if synonym != word and '_' not in synonym:  # Avoid multi-word synonyms and same word
                    new_words[i] = synonym

    return ' '.join(new_words)

original_text = "The quick, nimble fox swiftly dashed across the agile, sprightly meadow, evading the vigilant, watchful hound."
augmented_text = synonym_replacement(original_text)
print("Original Text: ", original_text)
print("Augmented Text: ", augmented_text)
