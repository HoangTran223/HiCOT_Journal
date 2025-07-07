from evaluations.IRBO import InvertedRBO  

def load_top_words(file_path):
    topics = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()  
            topics.append(words)  
    return topics

top_words_10 = load_top_words(top_words_10_path)

irbo = InvertedRBO(top_words_10)
irbo_score = irbo.score(topk=10, weight=0.9)
print(f"Inverted RBO (top 10): {irbo_score:.5f}")