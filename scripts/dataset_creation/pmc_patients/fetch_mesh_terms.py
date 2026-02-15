import json
import requests
import time
import xml.etree.ElementTree as ET

# Configuration
INPUT_FILE = "data/raw/pmc_patients/PMC-Patients-V2.json"
OUTPUT_FILE = "data/intermediate/pmc_patients/pmc_patients_with_mesh.jsonl"  # MeSH = Medical Subject Headings
BATCH_SIZE = 100

def fetch_mesh_batch(pmid_list):
    """Fetches MeSH terms for a list of PMIDs."""
    
    ids_str = ",".join(pmid_list)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={ids_str}&retmode=xml"
    
    try:
        response = requests.get(url)
        if response.status_code != 200: 
            print(f"Error (Status code: {response.status_code})")
            return {}
        
        root = ET.fromstring(response.content)
        results = {}
        
        # Default empty list for all requested PMIDs (fallback)
        for pmid in pmid_list:
            results[pmid] = []
            
        for article in root.findall(".//PubmedArticle"):
            pmid_node = article.find(".//PMID")
            if pmid_node is None: continue
            pmid = pmid_node.text
            
            terms = []
            for heading in article.findall(".//MeshHeading"):
                desc = heading.find("DescriptorName")
                if desc is not None:
                    terms.append(desc.text)
            results[pmid] = terms
        return results
    except Exception as e:
        print(f"Batch failed: {e}")
        return {}

def main():
    print("Reading input file...")
    entries = []
    with open(INPUT_FILE, 'r') as f:
        entries = json.load(f)

    print(f"Loaded {len(entries)} entries. Fetching metadata...")
    
    with open(OUTPUT_FILE, 'w') as f_out:
        for i in range(0, len(entries), BATCH_SIZE):
            batch = entries[i:i+BATCH_SIZE]
            pmids = [e['PMID'] for e in batch]
            
            # API Call
            mesh_map = fetch_mesh_batch(pmids)
            
            # Save results for this batch
            for entry in batch:
                pmid = entry['PMID']
                # Attach the terms to the object
                entry['mesh_terms'] = mesh_map.get(pmid, [])
                f_out.write(json.dumps(entry) + "\n")
            
            print(f"Progress: {min(i + BATCH_SIZE, len(entries))}/{len(entries)}")
            time.sleep(0.4) # 3 requests/sec limit (approx)

    print(f"Done. Checkpoint saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()