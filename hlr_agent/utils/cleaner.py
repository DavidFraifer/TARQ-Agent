"""
Tool name cleaner utility for HLR Agent.
Uses regex and similarity matching to clean tool names from LLM output.
"""

import re
from typing import Optional, List
from difflib import SequenceMatcher


def similarity(a: str, b: str) -> float:
    """Calculate similarity between two strings using SequenceMatcher."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def clean_tool_name(tool_name: str, available_tools: List[str], similarity_threshold: float = 0.6) -> Optional[str]:
    """
    Clean and match tool names using regex patterns and similarity matching.
    
    Args:
        tool_name: The potentially malformed tool name from LLM
        available_tools: List of valid tool names
        similarity_threshold: Minimum similarity score to consider a match (0.0-1.0)
    
    Returns:
        str: The closest matching tool name, or None if no good match found
    """
    if not tool_name or not available_tools:
        return None
    
    # Remove common prefixes/suffixes and clean the tool name
    cleaned_name = clean_tool_string(tool_name)
    
    # Direct match after cleaning
    if cleaned_name in available_tools:
        return cleaned_name
    
    # Find the most similar tool using different matching strategies
    best_match = None
    best_score = 0.0
    
    for available_tool in available_tools:
        # Strategy 1: Direct substring matching
        if cleaned_name in available_tool or available_tool in cleaned_name:
            substring_score = len(cleaned_name) / max(len(cleaned_name), len(available_tool))
            if substring_score > best_score and substring_score >= similarity_threshold:
                best_match = available_tool
                best_score = substring_score
        
        # Strategy 2: Sequence similarity
        seq_score = similarity(cleaned_name, available_tool)
        if seq_score > best_score and seq_score >= similarity_threshold:
            best_match = available_tool
            best_score = seq_score
        
        # Strategy 3: Regex pattern matching for common variations
        regex_score = regex_similarity(cleaned_name, available_tool)
        if regex_score > best_score and regex_score >= similarity_threshold:
            best_match = available_tool
            best_score = regex_score
    
    return best_match


def clean_tool_string(tool_name: str) -> str:
    """
    Clean tool string by removing common artifacts from LLM output.
    
    Args:
        tool_name: Raw tool name from LLM
        
    Returns:
        str: Cleaned tool name
    """
    if not tool_name:
        return ""
    
    # Convert to lowercase for processing
    cleaned = tool_name.lower().strip()
    
    # Remove common prefixes and suffixes
    patterns_to_remove = [
        r'^tool[_\-\s]*',           # "tool_", "tool-", "tool "
        r'^use[_\-\s]*',            # "use_", "use-", "use "
        r'^call[_\-\s]*',           # "call_", "call-", "call "
        r'^invoke[_\-\s]*',         # "invoke_", "invoke-", "invoke "
        r'[_\-\s]*tool$',           # "_tool", "-tool", " tool"
        r'[_\-\s]*api$',            # "_api", "-api", " api"
        r'[_\-\s]*service$',        # "_service", "-service", " service"
        r'[:;,\.\!\?]+$',           # trailing punctuation
        r'^[:;,\.\!\?]+',           # leading punctuation
    ]
    
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned)
    
    # Remove extra whitespace and special characters
    cleaned = re.sub(r'[^\w\-_]', '', cleaned)
    cleaned = re.sub(r'[-_\s]+', '', cleaned)
    
    return cleaned.strip()


def regex_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity using regex pattern matching for common variations.
    
    Args:
        name1: First name to compare
        name2: Second name to compare
        
    Returns:
        float: Similarity score (0.0-1.0)
    """
    if not name1 or not name2:
        return 0.0
    
    name1_clean = name1.lower()
    name2_clean = name2.lower()
    
    # Exact match
    if name1_clean == name2_clean:
        return 1.0
    
    # Check for common abbreviations and variations
    variations = {
        'gmail': ['email', 'mail', 'g-mail', 'googlemail'],
        'jira': ['ticket', 'issue', 'jira-ticket'],
        'sheets': ['spreadsheet', 'excel', 'googlesheets', 'g-sheets'],
        'drive': ['storage', 'file', 'googledrive', 'g-drive'],
        'calendar': ['cal', 'schedule', 'event', 'googlecalendar'],
        'slack': ['chat', 'message', 'notification', 'slack-message']
    }
    
    # Check if name1 matches any variations of name2
    if name2_clean in variations:
        for variation in variations[name2_clean]:
            if name1_clean == variation or variation in name1_clean:
                return 0.9  # High confidence for known variations
    
    # Check if name2 matches any variations of name1
    if name1_clean in variations:
        for variation in variations[name1_clean]:
            if name2_clean == variation or variation in name2_clean:
                return 0.9  # High confidence for known variations
    
    # Check for partial matches with common prefixes/suffixes
    common_prefixes = ['google', 'g', 'use', 'call', 'invoke']
    common_suffixes = ['api', 'tool', 'service', 'app']
    
    # Remove common prefixes/suffixes and check similarity
    name1_stripped = name1_clean
    name2_stripped = name2_clean
    
    for prefix in common_prefixes:
        name1_stripped = re.sub(f'^{prefix}[-_]?', '', name1_stripped)
        name2_stripped = re.sub(f'^{prefix}[-_]?', '', name2_stripped)
    
    for suffix in common_suffixes:
        name1_stripped = re.sub(f'[-_]?{suffix}$', '', name1_stripped)
        name2_stripped = re.sub(f'[-_]?{suffix}$', '', name2_stripped)
    
    if name1_stripped == name2_stripped and name1_stripped:
        return 0.8  # Good match after removing common affixes
    
    # Calculate character-level similarity for the stripped names
    if name1_stripped and name2_stripped:
        return similarity(name1_stripped, name2_stripped)
    
    return 0.0


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    available_tools = ['jira', 'gmail', 'sheets', 'drive', 'calendar', 'slack']
    
    test_cases = [
        "gmail:",
        "jira-ticket",
        "google-sheets",
        "email-tool",
        "slack_message",
        "drive_api",
        "calendar_service",
        "unknown_tool",
        "random_string"
    ]
    
    print("Tool Name Cleaner Test Results:")
    print("=" * 50)
    
    for test_case in test_cases:
        result = clean_tool_name(test_case, available_tools)
        print(f"'{test_case}' -> '{result}'")
