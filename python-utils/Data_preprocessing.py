#Data_preprocessing.py

class Preprocessing:
    def __init__(self):
        pass

    def remove_index(self, text):
        import re  # Importing the 're' module inside the function
        
        # Split the text into lines
        lines = text.split("\n")
        
        # Flag to detect if we are in the "Indice" section
        in_index = False
        cleaned_lines = []
        
        for line in lines:
            # Detect the start of the index section by looking for "Indice"
            if re.search(r'\bIndice\b', line, re.IGNORECASE):
                in_index = True
                continue  # Skip the "Indice" line itself
            
            # If we are in the index section, check if the line is part of the index
            if in_index:
                # Check for lines that contain a sentence followed by dots or a number
                if re.match(r'^.*\s*\.*\s*\d+\s*$', line.strip()):
                    continue  # Skip this line (it's part of the index)
                # Check for lines that only contain a number (page breaks)
                elif re.match(r'^\d+$', line) or re.match(r'^[\s]*$', line):
                    continue  # Skip page numbers or empty lines
                else:
                    # If we hit a line that doesn't match the index format, we're past the index section
                    in_index = False
            
            # Add the current line to the cleaned_lines if it's not part of the index
            cleaned_lines.append(line)
        
        # Recombine the cleaned lines into a single string
        return "\n".join(cleaned_lines)

    def index_dots_removal(self, text):
        import re
        # Split the text into individual lines
        lines = text.splitlines("\n")

        # Define the regex pattern to match lines with more than 10 dots and ending with a digit
        pattern = r"\.{10,}\s*"

        # Filter out lines that match the pattern
        filtered_lines = [line for line in lines if not re.search(pattern, line)]

        # Join the filtered lines back into a single string
        filtered_text = "\n".join(filtered_lines)
        return filtered_text

    def remove_remaining_indexes(self, text):
        import re
        pattern = r"(Obiettivi del manuale).*(\n\n\n1\. )"
        
        cleaned_text = re.sub(pattern, ' ', text, flags=re.DOTALL)
        
        return cleaned_text

    def remove_table_content(self, text):
        """
        Removes table-like content from the text, defined as sections starting with 'Es.'
        and containing at least two '|' symbols, which represent manually written tables.
        
        Args:
        - text (str): The input text from which table content needs to be removed.
        
        Returns:
        - str: The modified text with table-like content removed.
        """
        import re
        
        # The pattern looks for blocks that begin with 'Es.' and contain at least two '|' symbols.
        pattern = r'Es\..*?(?:\|.*?){2,}.*?(?=\n\n|\Z)'
        
        # Using re.sub to remove matching table-like blocks
        modified_text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # Also clean up any remaining multiple new lines created after table removal
        modified_text = re.sub(r'\n\s*\n+', '\n\n', modified_text)
        
        return modified_text
    
    def remove_remaining_table_from_text(self, text):
        import re
        # Define the regex pattern
        pattern = r"\n\ntabella[\s\S]*?\|.*\|\s*\n\n"
        
        # Use re.sub to substitute the matched table with an empty string
        cleaned_text = re.sub(pattern, '\n', text, flags=re.MULTILINE)
        
        return cleaned_text

    def remove_table_header(self, text):
        import re
        pattern = r"--+|===+"
        
        cleaned_text = re.sub(pattern, '', text, flags=re.MULTILINE)
        return cleaned_text

    def flatten_any_list(self, text):
        import re
        # Updated regex pattern to match:
        # 1. Bullet lists (e.g., *, -, •, \uf0a7)
        # 2. Numbered lists (e.g., 1., 2)
        # 3. Uppercase alphabetic markers (e.g., I, U, D)
        # 4. Lowercase alphabetic markers (e.g., a., b.)
        # 5. Mixed patterns (e.g., I Inserimento, 0 Inserimento/Aggiornamento)
        
        
        #list_pattern = r"(\n\s*[\*\-\•\–]|\n\s*\d+[\.\)]|\n\s*[a-zA-Z]{1}|\n\s*\d+|\uf0a7)\s+.+(?:\s+.+)*"
        # 6. Match nested numeration (e.g. 1.1.1., 2.30)
        # 7. Match number or alphabetic lists with ')' (e.g. 1) 2) or a) b) )
        # 8. Mixed number/letter list (e.g. 1a), 1b) or 1a. 1b. ) 
        list_pattern = r"(\n\s*[\*\-\•\–\▪]|\n\s*\d+[\.\)]|\n\s*(\d)*\w[\.\)]|\n\s*[a-zA-Z]{1}|\n\s*\d+|\n\s*(\d+[\.\)]?)+|\\uf0a7)\s+.+(?:\s+.+)*"
        

        # Function to replace the matched list items
        def replace_list_with_commas(match):
            import re
            # Get the matched list block
            list_block = match.group(0)
            
            # Remove the list markers and flatten the list
            # flattened_list = re.sub(r"(\n\s*[\*\-\•\–]|\n\s*\d+[\.\)]|\n\s*[a-zA-Z]{1}|\n\s*\d+|\uf0a7)\s+", ", ", list_block)
            flattened_list = re.sub(r"(\n\s*[\*\-\•\–]|\n\s*\d+[\.\)]|\n\s*(\d)*\w[\.\)]|\n\s*[a-zA-Z]{1}|\n\s*\d+|\n\s*(\d+[\.\)]?)+|\\uf0a7)\s+", ", ", list_block)
            
            flattened_list = re.sub(r"\n\s*", " ", flattened_list)  # Removes extra newlines within list items
            
            # Clean up spaces
            flattened_list = flattened_list.replace("  ", " ").strip()
            
            
            return flattened_list

        # Apply the transformation only to list items, leaving other text untouched
        normalized_text = re.sub(list_pattern, replace_list_with_commas, text)
        normalized_text = re.sub(r": ,", ": ", normalized_text)  # Fix for colon-space issues
        
        return normalized_text

    def remove_issue_date(self, text):
        import re
        # Define the regex pattern to match
        pattern = r'Data emissione (\d+\/){2}(\d+)'
        
        # Remove all matches of the pattern
        cleaned_text = re.sub(pattern, '', text)
        
        return cleaned_text
    
    def remove_footer(self, text):
            import re
            # Define the regex pattern to match
            pattern = r'\d{1,2}\/\d{1,2}\s\d{1,2}\/\d{1,2}\/\d{4}'
            
            # Remove all matches of the pattern
            cleaned_text = re.sub(pattern, '', text)
            
            return cleaned_text

    def normalize_whitespace(self, text):
        import re
        """
        Reduces multiple consecutive whitespace characters to a single space.

        Args:
        - text (str): The input text with excessive whitespace.

        Returns:
        - str: The text with reduced whitespace.
        """
        # Replace one or more whitespace characters with a single space
        return re.sub(r'\s+', ' ', text).strip()

    def clean_text_template(self, text):
        text = Preprocessing.remove_index(self, text)
        text = Preprocessing.index_dots_removal(self, text)
        text = Preprocessing.remove_remaining_indexes(self, text)
        text = Preprocessing.remove_table_content(self, text)
        text = Preprocessing.remove_remaining_table_from_text(self, text)
        text = Preprocessing.remove_table_header(self, text)
        text = Preprocessing.flatten_any_list(self, text)
        text = Preprocessing.remove_issue_date(self, text)
        #text = Preprocessing.normalize_whitespace(self, text)
        #text = Preprocessing.remove_footer(self,text)
        return text