import spacy
import tkinter as tk
from tkinter import scrolledtext, messagebox

nlp = spacy.load("ru_core_news_md")

def analyze_text():
    text = text_entry.get("1.0", tk.END).strip()

    if not text:
        messagebox.showwarning("Ошибка", "Введите текст для анализа.")
        return

    doc = nlp(text)

    result = "\nАнализ текста:\n"

    for token in doc:
        result += f"{token.text} -> Лемма: {token.lemma_}, Часть речи: {token.pos_}\n"

    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, result)
    output_text.config(state=tk.DISABLED)

def compare_words():
    word1 = word1_entry.get().strip()
    word2 = word2_entry.get().strip()

    if not word1 or not word2:
        messagebox.showwarning("Ошибка", "Введите два слова для сравнения.")
        return

    word1_doc = nlp(word1)
    word2_doc = nlp(word2)

    similarity = word1_doc.similarity(word2_doc)

    result = f"Семантическая близость между '{word1}' и '{word2}': {similarity:.2f}"

    output_text.config(state=tk.NORMAL)
    output_text.insert(tk.END, "\n" + result)
    output_text.config(state=tk.DISABLED)

root = tk.Tk()
root.title("Семантический анализ текста")

text_entry = tk.Text(root, height=10, width=60)
text_entry.pack()

tk.Button(root, text="Анализ текста", command=analyze_text).pack()

word1_entry = tk.Entry(root)
word1_entry.pack()

word2_entry = tk.Entry(root)
word2_entry.pack()

tk.Button(root, text="Сравнить слова", command=compare_words).pack()

output_text = scrolledtext.ScrolledText(root, height=15, width=60, state=tk.DISABLED)
output_text.pack()

root.mainloop()
