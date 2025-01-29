import matplotlib.pyplot as plt

labels = list(result.keys())
values = list(result.values())

# Erstellen des Histogramms
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, values, color='skyblue')

# Histogramm wird erzeugt
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  
        height,                             
        f'{int(height)}',                   
        ha='center',                        
        va='bottom',                        
        fontsize=9                          
    )

plt.xticks(rotation=90)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Histogram of Data Distribution')
plt.tight_layout()

plt.show()
