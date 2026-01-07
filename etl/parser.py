# etl/parser.py
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List

@dataclass
class PatchChunk:
    header: str    # Champion Name (e.g., "Aatrox")
    sub_header: str # Ability (e.g., "The Darkin Blade")
    content: str   # The change (e.g., "Damage increased...")
    patch_version: str
    
    def to_text(self):
        return f"Patch {self.patch_version} | {self.header} | {self.sub_header}: {self.content}"

class LoLPatchParser:
    def __init__(self, html_content: str, patch_version: str):
        self.soup = BeautifulSoup(html_content, 'html.parser')
        self.patch_version = patch_version

    def parse(self) -> List[PatchChunk]:
        chunks = []
        # Find the main container - Riot changes this ID often, so we target the wrapper
        article = self.soup.find('div', class_='style__Wrapper-sc-1h71jyd-0') # Common Riot Class
        
        # Fallback if class changed
        if not article:
             article = self.soup.find('div', id='patch-notes-container')

        if not article:
            return [] # Logic to handle failure

        current_champ = "General"
        current_ability = "Base Stats"

        # Looping through tags
        for tag in article.find_all(['h3', 'h4', 'li', 'p']):
            text = tag.get_text(strip=True)
            
            if tag.name == 'h3':
                current_champ = text
                current_ability = "Summary"
            elif tag.name == 'h4':
                current_ability = text
            elif tag.name in ['li', 'p']:
                if len(text) > 15: # Filter noise
                    chunks.append(PatchChunk(current_champ, current_ability, text, self.patch_version))
                    
        return chunks