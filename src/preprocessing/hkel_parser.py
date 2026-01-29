"""
HKEL XML Parser for Hong Kong e-Legislation

This module parses the HKEL XML format and extracts structured legal content
for the Legal RAG system.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from pathlib import Path
import json
import re


# XML Namespaces used in HKEL documents
NAMESPACES = {
    'hklm': 'http://www.xml.gov.hk/schemas/hklm/1.0',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'dcterms': 'http://purl.org/dc/terms/',
    'xhtml': 'http://www.w3.org/1999/xhtml'
}


@dataclass
class Definition:
    """Legal term definition"""
    name: str
    term_en: str
    term_zh: Optional[str] = None
    content: str = ""
    source_section: str = ""
    ordinance: str = ""


@dataclass
class CrossReference:
    """Reference to another legal provision"""
    href: str
    text: str
    source_id: str
    target_ordinance: Optional[str] = None
    target_section: Optional[str] = None


@dataclass
class LegalSection:
    """A section or subsection of an ordinance"""
    id: str
    name: str  # e.g., "s2", "s2_1"
    num: str   # e.g., "2.", "(1)"
    heading: Optional[str] = None
    content: str = ""
    status: str = "operational"  # operational, repealed
    effective_date: Optional[str] = None
    hierarchy_path: str = ""
    parent_id: Optional[str] = None
    children: List['LegalSection'] = field(default_factory=list)
    definitions: List[Definition] = field(default_factory=list)
    cross_references: List[CrossReference] = field(default_factory=list)
    raw_xml: str = ""


@dataclass
class LegalPart:
    """A part of an ordinance"""
    id: str
    name: str
    num: str
    heading: str
    sections: List[LegalSection] = field(default_factory=list)


@dataclass
class Ordinance:
    """Complete ordinance document"""
    doc_name: str        # e.g., "Cap. 344"
    doc_type: str        # "cap" or "instrument"
    doc_number: str      # e.g., "344"
    doc_status: str      # "In effect"
    short_title: str     # "Building Management Ordinance"
    long_title: str      # Full description
    effective_date: str
    parts: List[LegalPart] = field(default_factory=list)
    sections: List[LegalSection] = field(default_factory=list)  # Sections not in parts
    all_definitions: Dict[str, Definition] = field(default_factory=dict)
    all_cross_references: List[CrossReference] = field(default_factory=list)
    file_path: str = ""


class HKELParser:
    """Parser for Hong Kong e-Legislation XML format"""

    def __init__(self):
        self.ns = NAMESPACES

    def parse_file(self, file_path: Path) -> Ordinance:
        """Parse a single HKEL XML file"""
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Determine root element type (ordinance or subLeg)
        root_tag = root.tag.replace(f'{{{self.ns["hklm"]}}}', '')

        ordinance = Ordinance(
            doc_name="",
            doc_type="",
            doc_number="",
            doc_status="",
            short_title="",
            long_title="",
            effective_date="",
            file_path=str(file_path)
        )

        # Parse metadata
        self._parse_metadata(root, ordinance)

        # Parse main content
        main = root.find('hklm:main', self.ns)
        if main is not None:
            self._parse_main_content(main, ordinance)

        return ordinance

    def _parse_metadata(self, root: ET.Element, ordinance: Ordinance):
        """Extract document metadata"""
        meta = root.find('hklm:meta', self.ns)
        if meta is None:
            return

        ordinance.doc_name = self._get_text(meta, 'hklm:docName')
        ordinance.doc_type = self._get_text(meta, 'hklm:docType')
        ordinance.doc_number = self._get_text(meta, 'hklm:docNumber')
        ordinance.doc_status = self._get_text(meta, 'hklm:docStatus')
        ordinance.effective_date = self._get_text(meta, 'dc:date')

    def _parse_main_content(self, main: ET.Element, ordinance: Ordinance):
        """Parse the main content of the ordinance"""

        # Extract long title
        long_title = main.find('hklm:longTitle', self.ns)
        if long_title is not None:
            ordinance.long_title = self._extract_text_content(long_title)

            # Extract short title if present
            short_title_elem = long_title.find('.//hklm:shortTitle', self.ns)
            if short_title_elem is not None:
                ordinance.short_title = self._extract_text_content(short_title_elem)

        # Check for docTitle (used in subsidiary legislation)
        doc_title = main.find('hklm:docTitle', self.ns)
        if doc_title is not None and not ordinance.short_title:
            ordinance.short_title = self._extract_text_content(doc_title)

        # Parse parts
        for part_elem in main.findall('hklm:part', self.ns):
            part = self._parse_part(part_elem, ordinance)
            ordinance.parts.append(part)

        # Parse sections not within parts
        for section_elem in main.findall('hklm:section', self.ns):
            section = self._parse_section(section_elem, ordinance.doc_name, "")
            ordinance.sections.append(section)

            # Collect definitions and cross-references
            for defn in section.definitions:
                ordinance.all_definitions[defn.name] = defn
            ordinance.all_cross_references.extend(section.cross_references)

    def _parse_part(self, part_elem: ET.Element, ordinance: Ordinance) -> LegalPart:
        """Parse a part element"""
        part = LegalPart(
            id=part_elem.get('id', ''),
            name=part_elem.get('name', ''),
            num=self._get_text(part_elem, 'hklm:num'),
            heading=self._get_text(part_elem, 'hklm:heading')
        )

        hierarchy_prefix = f"{ordinance.doc_name} > {part.heading or part.num}"

        # Parse sections within this part
        for section_elem in part_elem.findall('hklm:section', self.ns):
            section = self._parse_section(section_elem, ordinance.doc_name, hierarchy_prefix)
            part.sections.append(section)

            # Collect definitions and cross-references at ordinance level
            for defn in section.definitions:
                defn.ordinance = ordinance.doc_name
                ordinance.all_definitions[defn.name] = defn
            ordinance.all_cross_references.extend(section.cross_references)

        return part

    def _parse_section(self, section_elem: ET.Element, ordinance_name: str,
                       hierarchy_prefix: str) -> LegalSection:
        """Parse a section element"""
        section = LegalSection(
            id=section_elem.get('id', ''),
            name=section_elem.get('name', ''),
            num=self._get_text(section_elem, 'hklm:num'),
            heading=self._get_text(section_elem, 'hklm:heading'),
            status=section_elem.get('status', 'operational'),
            effective_date=section_elem.get('startPeriod'),
            raw_xml=ET.tostring(section_elem, encoding='unicode')
        )

        section.hierarchy_path = f"{hierarchy_prefix} > {section.heading or section.num}".strip(' >')

        # Extract content (excluding subsections)
        section.content = self._extract_section_content(section_elem)

        # Parse definitions
        for def_elem in section_elem.findall('.//hklm:def', self.ns):
            defn = self._parse_definition(def_elem, section.name)
            section.definitions.append(defn)

        # Parse cross-references
        for ref_elem in section_elem.findall('.//hklm:ref', self.ns):
            ref = self._parse_cross_reference(ref_elem, section.id)
            section.cross_references.append(ref)

        # Parse subsections
        for subsection_elem in section_elem.findall('hklm:subsection', self.ns):
            subsection = self._parse_subsection(subsection_elem, section)
            section.children.append(subsection)

        return section

    def _parse_subsection(self, elem: ET.Element, parent: LegalSection) -> LegalSection:
        """Parse a subsection element"""
        subsection = LegalSection(
            id=elem.get('id', ''),
            name=elem.get('name', ''),
            num=self._get_text(elem, 'hklm:num'),
            status=elem.get('status', 'operational'),
            effective_date=elem.get('startPeriod'),
            parent_id=parent.id,
            hierarchy_path=f"{parent.hierarchy_path} > {self._get_text(elem, 'hklm:num')}"
        )

        subsection.content = self._extract_section_content(elem)

        # Parse definitions in subsection
        for def_elem in elem.findall('.//hklm:def', self.ns):
            defn = self._parse_definition(def_elem, subsection.name)
            subsection.definitions.append(defn)

        # Parse cross-references
        for ref_elem in elem.findall('.//hklm:ref', self.ns):
            ref = self._parse_cross_reference(ref_elem, subsection.id)
            subsection.cross_references.append(ref)

        # Parse paragraphs as children
        for para_elem in elem.findall('hklm:paragraph', self.ns):
            para = self._parse_paragraph(para_elem, subsection)
            subsection.children.append(para)

        return subsection

    def _parse_paragraph(self, elem: ET.Element, parent: LegalSection) -> LegalSection:
        """Parse a paragraph element (treated as a section for consistency)"""
        para = LegalSection(
            id=elem.get('id', ''),
            name=elem.get('name', ''),
            num=self._get_text(elem, 'hklm:num'),
            status='operational',
            parent_id=parent.id,
            hierarchy_path=f"{parent.hierarchy_path} > {self._get_text(elem, 'hklm:num')}"
        )

        para.content = self._extract_section_content(elem)

        # Parse cross-references
        for ref_elem in elem.findall('.//hklm:ref', self.ns):
            ref = self._parse_cross_reference(ref_elem, para.id)
            para.cross_references.append(ref)

        return para

    def _parse_definition(self, def_elem: ET.Element, source_section: str) -> Definition:
        """Parse a definition element"""
        defn = Definition(
            name=def_elem.get('name', ''),
            term_en='',
            source_section=source_section
        )

        # Extract English term
        term_elem = def_elem.find('hklm:term', self.ns)
        if term_elem is not None:
            defn.term_en = self._extract_text_content(term_elem)

        # Extract Chinese term
        for term_elem in def_elem.findall('.//hklm:term[@xml:lang]', self.ns):
            lang = term_elem.get('{http://www.w3.org/XML/1998/namespace}lang', '')
            if 'zh' in lang:
                defn.term_zh = self._extract_text_content(term_elem)
                break

        # Extract definition content
        defn.content = self._extract_text_content(def_elem)

        return defn

    def _parse_cross_reference(self, ref_elem: ET.Element, source_id: str) -> CrossReference:
        """Parse a cross-reference element"""
        href = ref_elem.get('href', '')

        ref = CrossReference(
            href=href,
            text=self._extract_text_content(ref_elem),
            source_id=source_id
        )

        # Parse href to extract ordinance and section
        if href:
            # Pattern: /hk/cap344 or /hk/cap344/s3
            match = re.match(r'/hk/(cap\d+[A-Z]?)(?:/(.+))?', href)
            if match:
                ref.target_ordinance = f"Cap. {match.group(1).replace('cap', '')}"
                ref.target_section = match.group(2)

        return ref

    def _get_text(self, elem: ET.Element, path: str) -> str:
        """Get text content of a child element"""
        child = elem.find(path, self.ns)
        if child is not None:
            return self._extract_text_content(child)
        return ""

    def _extract_text_content(self, elem: ET.Element) -> str:
        """Extract all text content from an element, including nested elements"""
        texts = []

        if elem.text:
            texts.append(elem.text.strip())

        for child in elem:
            child_text = self._extract_text_content(child)
            if child_text:
                texts.append(child_text)
            if child.tail:
                texts.append(child.tail.strip())

        return ' '.join(filter(None, texts))

    def _extract_section_content(self, elem: ET.Element) -> str:
        """Extract content from a section, excluding nested sections"""
        texts = []

        # Elements to skip (they are parsed separately)
        skip_tags = {'subsection', 'section', 'part'}

        if elem.text:
            texts.append(elem.text.strip())

        for child in elem:
            tag = child.tag.replace(f'{{{self.ns["hklm"]}}}', '')

            if tag not in skip_tags:
                child_text = self._extract_text_content(child)
                if child_text:
                    texts.append(child_text)

            if child.tail:
                texts.append(child.tail.strip())

        return ' '.join(filter(None, texts))


def parse_ordinance_directory(directory: Path) -> List[Ordinance]:
    """Parse all ordinances in a directory"""
    parser = HKELParser()
    ordinances = []

    for xml_file in directory.rglob('*.xml'):
        try:
            ordinance = parser.parse_file(xml_file)
            ordinances.append(ordinance)
            print(f"Parsed: {ordinance.doc_name} - {ordinance.short_title}")
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")

    return ordinances


if __name__ == "__main__":
    # Test parsing
    data_dir = Path("/home/user/legal_advisor_pytorch/data/property_owner_ordinances")
    ordinances = parse_ordinance_directory(data_dir)

    print(f"\nTotal ordinances parsed: {len(ordinances)}")

    # Print summary
    for ord in ordinances[:5]:
        print(f"\n{ord.doc_name}: {ord.short_title}")
        print(f"  Parts: {len(ord.parts)}")
        print(f"  Definitions: {len(ord.all_definitions)}")
        print(f"  Cross-references: {len(ord.all_cross_references)}")
