from nltk.tokenize import sent_tokenize
import string
from typing import List, Dict, Tuple
import json

def extract_relevant_info_serper(search_results):
    """
    Extract relevant information from Google Serper search results.

    Args:
        search_results (dict): JSON response from the Google Serper API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    if 'organic' in search_results:
        for i, result in enumerate(search_results['organic']):
            # Try to extract domain for site_name, or leave empty
            site_name = ''
            try:
                site_name = urlparse(result.get('link', '')).netloc
            except Exception:
                pass

            info = {
                'id': i + 1,
                'title': result.get('title', ''),
                'url': result.get('link', ''),
                'site_name': site_name, # Serper doesn't directly provide siteName, try to parse from URL
                'date': result.get('date', ''), # Serper might not always provide date
                'snippet': result.get('snippet', ''),
            }
            useful_info.append(info)
    return useful_info


def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 3000) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:100000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        # sentences = re.split(r'(?<=[.!?]) +', full_text)  # Split sentences using regex, supporting ., !, ? endings
        sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
          
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            # if end_index - start_index < 2 * context_chars:
            #     end_index = min(len(full_text), start_index + 2 * context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"
    

def json_loads(text: str) -> dict:
    text = text.strip('\n')
    if text.startswith('```') and text.endswith('\n```'):
        text = '\n'.join(text.split('\n')[1:-1])
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError as json_err:
        try:
            return json5.loads(text)
        except ValueError:
            raise json_err