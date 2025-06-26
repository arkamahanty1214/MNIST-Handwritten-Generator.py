import numpy as np
import re
import os
from collections import defaultdict
import pickle

class TextGenerator:
    def __init__(self, seq_length=4):
        self.seq_length = seq_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.smoothing = 0.01

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  
        text = re.sub(r'\s+', ' ', text).strip()  
        return text

    def train(self, text):
        text = self.clean_text(text)
        vocab = sorted(list(set(text)))
        self.char_to_idx = {c: i for i, c in enumerate(vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(vocab)}
        
        for i in range(len(text) - self.seq_length):
            context = text[i:i+self.seq_length]
            next_char = text[i+self.seq_length]
            self.transitions[context][next_char] += 1

        for context in self.transitions:
            total = sum(self.transitions[context].values()) + self.smoothing*len(vocab)
            for char in vocab:
                self.transitions[context][char] = (self.transitions[context].get(char, 0) + self.smoothing) / total

    def generate(self, seed, length=100, temperature=1.0):
        seed = self.clean_text(seed)[-self.seq_length:] 
        result = seed
        
        for _ in range(length):
            context = result[-self.seq_length:]
            if context not in self.transitions:
                break  
            
            chars, probs = zip(*self.transitions[context].items())
            probs = np.array(probs) ** (1/temperature)
            probs = probs / probs.sum()
            
            next_char = np.random.choice(chars, p=probs)
            result += next_char
            
        return result

    def save(self, filepath):
         with open(filepath, 'wb') as f:
            pickle.dump({
                'seq_length': self.seq_length,
                'char_to_idx': self.char_to_idx,
                'transitions': dict(self.transitions)
            }, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        generator = cls(data['seq_length'])
        generator.char_to_idx = data['char_to_idx']
        generator.idx_to_char = {v:k for k,v in data['char_to_idx'].items()}
        generator.transitions = defaultdict(lambda: defaultdict(float), data['transitions'])
        return generator

def load_training_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return None

def main():
    
    MODEL_FILE = 'text_gen_model.pkl'
    TRAINING_FILE = 'training_text.txt'
    SEQ_LENGTH = 5 
    
    SAMPLE_TEXT = """
    The quick brown fox jumps over the lazy dog.
    Pack my box with five dozen liquor jugs.
    How vexingly quick daft zebras jump!
    A quick movement of the enemy will jeopardize six gunboats.
    """ * 5  

    
    generator = TextGenerator(seq_length=SEQ_LENGTH)
    
    
    text = load_training_data(TRAINING_FILE) or SAMPLE_TEXT
    
    # Train the model
    print("Training model...")
    generator.train(text)
    
    # Save the model
    generator.save(MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    
    # Generate sample text
    print("\nText Generation Examples")
    
    seeds = ["quick", "fox j", "lazy", "zebra"]
    for seed in seeds:
        print(f"\nSeed: '{seed}'")
        for temp in [0.5, 1.0, 1.5, 2.0]:
            print(f"Temp {temp}: {generator.generate(seed, length=70, temperature=temp)}")

if __name__ == "__main__":
    main()