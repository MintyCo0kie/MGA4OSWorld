import os
import fitz  # PyMuPDF

def check_files_and_contents(directory, filenames, strings_to_check):
    results = []
    for filename, string_to_check in zip(filenames, strings_to_check):
        file_path = os.path.join(directory, filename)
        if not os.path.isfile(file_path):
            results.append(-1) 
            continue
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            if string_to_check in text:
                results.append(1)  
            else:
                results.append(0)  
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            results.append(0)  
        finally:
            if doc:
                doc.close()
    return results


directory = "/home/user/Documents/Blog"  
filenames = ["LLM Powered Autonomous Agents.pdf", "Thinking about High-Quality Human Data.pdf"]
strings_to_check = ["LLM Powered Autonomous Agents", "Thinking about High-Quality Human Data"]  


results = check_files_and_contents(directory, filenames, strings_to_check)
print(results)
