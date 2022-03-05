
webxray_json_path="domain_owners.json"


def load_webxray_domain_ownership_list(webxray_json_path):
    webxray_list = json.loads(open(webxray_json_path).read())
    domain_orgs = {}  # `domain name` to (org) `id` mapping
    parent_orgs = {}  # (org) `id` to `parent_id` mapping
    org_names = {}  # (org) `id` to `owner_name` mapping
    domain_owners = {}  # domain to topmost parent organization name mapping

    for data in webxray_list:
        org_names[data["id"]] = data["owner_name"]
        if data["parent_id"]:
            parent_orgs[data["id"]] = data["parent_id"]
        for domain in data["domains"]:
            domain_orgs[domain] = data["id"]


    for domain, org_id in domain_orgs.iteritems():
        domain_owners[domain] = org_names[get_topmost_parent(org_id, parent_orgs)]
    return domain_owners


# In[4]:

def get_topmost_parent(org_id, parent_orgs):
    """Walk up the parent organizations dict."""
    while org_id in parent_orgs:
        org_id = parent_orgs[org_id]  # get the parent's id
    return org_id


# ### Load domain ownership mapping

# In[5]:

# You should download `domain_owners.json` from the following link
# https://github.com/timlib/webXray_Domain_Owner_List/blob/master/domain_owners.json
if not os.path.exists('domain_owners.json'):
    url = 'https://raw.githubusercontent.com/timlib/webXray_Domain_Owner_List/master/domain_owners.json'
    os.system('wget {}'.format(url))
    if not os.path.exists('domain_owners.json'):
        exit('You should download domain_owners.json')

domain_owners = load_webxray_domain_ownership_list("domain_owners.json")
