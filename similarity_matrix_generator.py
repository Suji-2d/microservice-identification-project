import os
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt


# Preprocessing
def preprocess_text(text):

    java_keywords = ['abstract', 'continue','for','new','switch', 'assert','default','goto','package','synchronized','boolean','do','if','private','this',
'break','double','implements','protected','throw',
'byte','else','import','public','throws',
'case','enum','instanceof','return','transient',
'catch','extends','int','short','try',
'char','final','interface','static','void',
'class','finally','long','strictfp','volatile',
'const','float','native','super','while']
    
    # Remove key words
    for word in java_keywords:
        text = re.sub(word,'',text)
    
    # Remove empty lines and comments
    text = re.sub(r'\/\/.*', '', text)
    text = re.sub(r'\/\*.*?\*\/', '', text, flags=re.DOTALL)
    
    # Remove camel case
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    

    # Tokenize and remove punctuation
    tokens = nltk.word_tokenize(text)


    tokens = [word.lower() for word in tokens]
    tokens = [word.strip() for word in tokens]
    tokens = [word for word in tokens if word.isalnum()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    
    return ' '.join(tokens)


# Collect Java files
java_folder = '/Users/sujith/eclipse-msc/acmeair/petclinic'
java_files = [file for file in os.listdir(java_folder) if file.endswith('.java')]

# Preprocess and collect processed texts
processed_texts = []
for java_file in java_files:
    with open(os.path.join(java_folder, java_file), 'r') as file:
        content = file.read()
        processed_content = preprocess_text(content)
        processed_texts.append(processed_content)

# Combine all processed texts into a single string
all_processed_text = ' '.join(processed_texts)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=800, background_color='white',collocations=False).generate(all_processed_text)

# Display the word cloud using matplotlib
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# TF-IDF calculation
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

tfidf_df.round(3).to_csv('tfidf_df.csv')

java_name = [j.replace(".java","") for j in java_files]

# Calculate similarity
similarity_matrix = cosine_similarity(tfidf_matrix)
simetic_similiraty_df = pd.DataFrame(similarity_matrix, columns = java_name, index = java_name)

# Perform hierarchical clustering
linkage_matrix = linkage(similarity_matrix, method='average')  # You can use other linkage methods as well

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=java_name)
plt.xlabel("Java Class")
plt.ylabel("Distance")
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

# Predict clusters using a chosen distance threshold
threshold = 1.35 # Adjust this threshold based on your data and dendrogram visualization
clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')

clusters = [int(c) for c in clusters]
# Plot the results
plt.scatter(clusters,java_name, c=clusters, cmap='viridis', s=50)
plt.xlabel('microservice')
plt.ylabel('Class Name')
plt.show()

# Calculate silhouette score
silhouette_avg = silhouette_score(similarity_matrix, clusters)
print("Silhouette Score:", silhouette_avg)

# Calculate Calinski-Harabasz index
ch_score = calinski_harabasz_score(similarity_matrix, clusters)
print("Calinski-Harabasz Index:", ch_score)

# Calculate Davies-Bouldin index
db_score = davies_bouldin_score(similarity_matrix, clusters)
print("Davies-Bouldin Index:", db_score)

# A higher silhouette score indicates better-defined clusters.
# A higher Calinski-Harabasz index suggests better separation between clusters.
# A lower Davies-Bouldin index indicates more distinct clusters.

# Combined approach

# Graph mat to similarity matrix
adj_matrix = pd.read_csv('graph_mat.csv')



adj_matrix.drop(['Unnamed: 0'],axis=1,inplace=True)
adj_matrix.set_index(adj_matrix.columns,inplace=True)
incoming_calls = np.sum(adj_matrix, axis=1)

print(sum(incoming_calls))
adj_matrix_np = adj_matrix.to_numpy()

sim_matrix = np.zeros(adj_matrix_np.shape)

num_nodes = adj_matrix.shape[0]

# Add edges with weights
for i in range(num_nodes):
    for j in range(num_nodes):
      cj=ci=0
      if incoming_calls[i]!=0: cj = adj_matrix_np[i,j]/incoming_calls[i]
      if incoming_calls[j]!=0: ci = adj_matrix_np[j,i]/incoming_calls[j]
      sim_matrix[i,j]=sim_matrix[j,i] = (cj+ci)*0.5 if i!=j else cj

# sim_matrix_norm = (sim_matrix-np.min(sim_matrix))/(np.max(sim_matrix)-np.min(sim_matrix))

call_similarity_df = pd.DataFrame(sim_matrix, columns=adj_matrix.index, index=adj_matrix.index)

print(call_similarity_df)

combie_similarity_df = simetic_similiraty_df.copy()

for index in adj_matrix.index:
    for col in adj_matrix.index:
       # print(index,col,call_similarity_df.loc[index][col])
        combie_similarity_df.loc[index][col] = simetic_similiraty_df.loc[index][col]*0.2 + call_similarity_df.loc[index][col]*1

#print(combie_similarity_df)

call_similarity_df.round(3).to_csv('call_similarity.csv')

simetic_similiraty_df.round(3).to_csv('simetic_similiraty.csv')

combie_similarity_df.round(3).to_csv('combie_similarity.csv')


