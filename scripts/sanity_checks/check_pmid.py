import requests
import xml.etree.ElementTree as ET

def get_mesh_terms(pmid):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
    response = requests.get(url)
    
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        mesh_terms = []
        for heading in root.findall(".//MeshHeading"):
            descriptor = heading.find("DescriptorName")
            if descriptor is not None:
                mesh_terms.append(descriptor.text)
        return mesh_terms
    else:
        return f"Error: {response.status_code}"

pmid = "15268761"
print(f"Checking PMID: {pmid}")
terms = get_mesh_terms(pmid)
print("Found MeSH Terms:", terms)
