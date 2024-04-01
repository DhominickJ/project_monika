# Import necessary libraries
import math

# Define the compute_tf function
def compute_tf(word_dict, list_of_words):
    tf_dict = {}
    word_count = len(list_of_words)
    for word, count in word_dict.items():
        tf_dict[word] = count / float(word_count)
    return tf_dict

# Define the compute_idf function
def compute_idf(doc_list):
    idf_dict = {}
    n = len(doc_list)
    
    idf_dict = dict.fromkeys(doc_list[0].keys(), 0)
    for doc in doc_list:
        for word, val in doc.items():
            if val > 0:
                idf_dict[word] += 1
    
    for word, val in idf_dict.items():
        idf_dict[word] = math.log(n / float(val + 1e-10))
         
    return idf_dict

# Define the compute_tf_idf function
def compute_tf_idf(tf_list_of_words, idfs):
    tfidf = {}
    for word, val in tf_list_of_words.items():
        tfidf[word] = val * idfs[word]
    return tfidf

def quick_sort(distance_list):
    if len(distance_list) <= 1:
        return distance_list
    else:
        # The pivot is the starting point of the list and the rest of the list is the pivot
        pivot = distance_list[0]
        less_than_pivot = [x for x in distance_list[1:] if x <= pivot]
        greater_than_pivot = [x for x in distance_list[1:] if x > pivot]
        return quick_sort(less_than_pivot) + [pivot] + quick_sort(greater_than_pivot)


# Define the knn_classifier function referenced from: https://github.com/CihanBosnali/K-Nearest-Neighbors-without-ML-libraries/blob/master/KNN-without-ML-libraries.ipynb
def knn_classifier(compute_tf_idf, tf_list_of_words, idfs, labels, test_sentence, k):
    # Store the distance in the array
    distances = []
    # Get the number of documents
    numberOfDocuments = len(tf_list_of_words)
    # Split the test sentence into a list of words
    test_list_of_words = test_sentence.split(" ")
    # Create a dictionary to hold the word count
    test_word_dict = dict.fromkeys(word_set, 0)
    # Count the words in the test sentence
    for word in test_list_of_words:
        if word in word_set:
            test_word_dict[word] += 1
            
    # Compute the term frequency for the test sentence and compute the tf-idf for the test sentence
    tf_test_list_of_words = compute_tf(test_word_dict, test_list_of_words)
    tfidf_test_list_of_words = compute_tf_idf(tf_test_list_of_words, idfs)

    for i in range(numberOfDocuments):
        # Compute the tf-idf values for the current document
        tfidf_current = compute_tf_idf(tf_list_of_words[i], idfs)
        distance = 0
        for word in tfidf_current.keys(): #Keys here are the individual words and the results is found in the number of the iteration of how frequent a word is
        # Add the squared difference of the tf-idf values
            distance += (tfidf_current.get(word, 0) - tfidf_test_list_of_words.get(word, 0))**2
        # Take the square root of the distance and append the distance and label towor
        distance = math.sqrt(distance)
        distances.append((distance, labels[i]))
    # Sort the list by the distance in ascending order
    distances = quick_sort(distances)
    # Get the k closest documents
    k_closest = distances[:k]
    # Get the labels of the k closest documents
    k_labels = [label for distance, label in k_closest]
    # Count the frequency of each label
    label_counts = {}
    for label in k_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    # Basically, if the count of an object with the more number of words is found, the label will change to the greater number of words that could be found on the sentence
    max_count = 0
    max_label = None
    for label, count in label_counts.items():
        if count > max_count:
            max_count = count
            max_label = label
    # Return the label of the test sentence
    return max_label

# Defining the value of k
k = 3

# Define the test documents
happy = "I am happy today, the world is a beautiful world!"
sad = "The world is a grim place, I wouldn't want to be there."
neutral = "The world is a neutral place, however I feel secured."

# Currently, error for keys not found or added in the dictionary
# test_sentence = "I am happy today, the world is a beautiful world and it is something I would want to see." # Happy
test_sentence = "The world is a grim place, I wouldn't want to be there." # Sad

# Split the documents into list of words
happy_list_of_words = happy.split(" ")
sad_list_of_words = sad.split(" ")
neutral_list_of_words = neutral.split(" ")

# Create a set of unique words 
word_set = set(happy_list_of_words).union(set(sad_list_of_words)).union(set(neutral_list_of_words))

# Create dictionaries to hold the word count
happy_word_dict = dict.fromkeys(word_set, 0) 
sad_word_dict = dict.fromkeys(word_set, 0)
neutral_word_dict = dict.fromkeys(word_set, 0)

# Count the words in each document
for word in happy_list_of_words:
    happy_word_dict[word]+=1
    
for word in sad_list_of_words:
    sad_word_dict[word]+=1

for word in neutral_list_of_words:
    neutral_word_dict[word]+=1

# Compute the term frequency for each document
happy_tf_list_of_words = compute_tf(happy_word_dict, happy_list_of_words)
sad_tf_list_of_words = compute_tf(sad_word_dict, sad_list_of_words)
neutral_tf_list_of_words = compute_tf(neutral_word_dict, neutral_list_of_words)

# Compute the inverse document frequency for the documents
idfs = compute_idf([happy_word_dict, sad_word_dict, neutral_word_dict])

# Compute the tf-idf for each document
happy_tfidf_list_of_words = compute_tf_idf(happy_tf_list_of_words, idfs)
sad_tfidf_list_of_words = compute_tf_idf(sad_tf_list_of_words, idfs)
neutral_tfidf_list_of_words = compute_tf_idf(neutral_tf_list_of_words, idfs)

# Define the labels for the documents
labels = ["Happy", "Sad", 'Neutral']

# Test the knn_classifier function
print(knn_classifier(compute_tf_idf, [happy_tfidf_list_of_words, sad_tfidf_list_of_words, neutral_tfidf_list_of_words], idfs, labels, test_sentence, k))
