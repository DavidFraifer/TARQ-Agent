from typing import List


class DSLParser:
    """Handles parsing of text DSL into executable flow structures"""
    
    def parse_text_dsl(self, text: str) -> List:
        """Parse text DSL into array format for execution"""
        lines = [line.rstrip() for line in text.strip().split('\n') if line.strip()]
        flow, i = [], 0
        
        while i < len(lines):
            line = lines[i]
            i += 1
            
            simple = self._parse_simple_command(line)
            if simple:
                flow.append(simple)
                continue
                    
            if line.strip().upper().startswith('IF'):
                condition = line.strip()[2:].strip()
                if condition.startswith('(') and condition.endswith(')'):
                    condition = condition[1:-1].strip()
                then_block, else_block, i = self._parse_conditional_block(lines, i, 2)
                flow.append(["IF", condition, then_block, else_block])
                
            elif line.strip().startswith('WHILE '):
                condition = line.strip()[6:].strip()
                body_block, i = self._parse_while_block(lines, i)
                flow.append(["WHILE", condition, body_block])
        
        return flow

    def _parse_simple_command(self, line: str):
        """Parse a single-line basic command (A/F/W/STOP). Returns list or None."""
        s = line.strip()
        if s.startswith('W '): return ["WAIT", int(s.split()[1])]
        if s.startswith('F '): return ["F", s.split()[1]]
        if s.startswith('A '):
            tool_part = s.split()[1]
            # Validate that A command only contains tool name, no parameters
            if '(' in tool_part or ')' in tool_part:
                raise ValueError(f"DSL Syntax Error: Action command 'A {tool_part}' contains parameters. Use only tool name: 'A {tool_part.split('(')[0]}'")
            return ["A", tool_part]
        if s == 'STOP': return ["STOP"]
        return None
    
    def _parse_conditional_block(self, lines: List[str], start_idx: int, expected_indent: int = 2):
        """Parse IF/ELSEIF/ELSE/ENDIF block"""
        then_block, else_block = [], []
        current_block = then_block
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            i += 1
            
            stripped = line.strip()
            if stripped == 'ENDIF':
                break
            elif stripped.upper().startswith('ELSEIF') or stripped == 'ELSE':
                current_block = else_block
                if stripped.upper().startswith('ELSEIF'):
                    condition = stripped[6:].strip()
                    if condition.startswith('(') and condition.endswith(')'):
                        condition = condition[1:-1].strip()
                    nested_then, nested_else, i = self._parse_conditional_block(lines, i, expected_indent)
                    else_block.append(["IF", condition, nested_then, nested_else])
                    break
                continue
            elif line.startswith(' ' * expected_indent):
                line_content = line[expected_indent:]
                
                if line_content.upper().startswith('IF'):
                    condition = line_content[2:].strip()
                    if condition.startswith('(') and condition.endswith(')'):
                        condition = condition[1:-1].strip()
                    nested_then, nested_else, new_i = self._parse_conditional_block(lines, i, expected_indent + 2)
                    current_block.append(["IF", condition, nested_then, nested_else])
                    i = new_i
                else:
                    cmd = self._parse_simple_command(line_content)
                    if cmd:
                        current_block.append(cmd)
        
        return then_block, else_block, i
    
    def _parse_while_block(self, lines: List[str], start_idx: int):
        """Parse WHILE/ENDWHILE block"""
        body_block = []
        i = start_idx
        expected_indent = 4
        
        while i < len(lines):
            line = lines[i]
            i += 1
            
            if line.strip() == 'ENDWHILE':
                break
            elif line.startswith(' ' * expected_indent):
                line_content = line[expected_indent:]
                if line_content.startswith('IF '):
                    condition = line_content[3:].strip()
                    if condition.startswith('(') and condition.endswith(')'):
                        condition = condition[1:-1].strip()
                    then_block, else_block, new_i = self._parse_conditional_block(lines, i, expected_indent + 4)
                    body_block.append(["IF", condition, then_block, else_block])
                    i = new_i
                else:
                    cmd = self._parse_simple_command(line_content)
                    if cmd:
                        body_block.append(cmd)
        
        return body_block, i

    def print_flow_structure(self, flow):
        """Print flow structure for debugging"""
        def print_item(flow_item, indent=0):
            spaces = "  " * indent
            if isinstance(flow_item, list) and len(flow_item) > 0:
                if flow_item[0] == "WHILE":
                    print(f"{spaces}WHILE {flow_item[1]}")
                    for sub_item in flow_item[2]:
                        print_item(sub_item, indent + 1)
                    print(f"{spaces}ENDWHILE")
                elif flow_item[0] == "IF":
                    print(f"{spaces}IF {flow_item[1]}")
                    for sub_item in flow_item[2]:
                        print_item(sub_item, indent + 1)
                    if len(flow_item) > 3 and flow_item[3]:
                        print(f"{spaces}ELSE")
                        for sub_item in flow_item[3]:
                            print_item(sub_item, indent + 1)
                    print(f"{spaces}ENDIF")
                else:
                    print(f"{spaces}{' '.join(str(x) for x in flow_item)}")
            else:
                print(f"{spaces}{flow_item}")
        
        if flow:
            for item in flow:
                print_item(item)
        else:
            print("Empty flow")
