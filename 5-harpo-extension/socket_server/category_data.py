import json
import os.path

category_list = ["Computers, Tablets & Accessories", "Audio", "TVs & Entertainment", "Cameras & Optics", "Cell Phones & Accessories", "Smart Electronics", "Business & Home Security Electronics", "Batteries & Power", "Telephones & Accessories", "Vehicle Audio & Electronics", "Drones, RC & Electronic Toys", "GPS, Navigation & Travel", "Electronic Components, Tools & Supplies", "Kids' Electronics", "Office Electronics", "Electronics Care", "Sports & Outdoor Electronics", "Kitchen & Dining", "Appliances", "Bedding & Bath", "Lawn & Garden", "Lighting", "Home Decor", "Wall Art & Decor", "Furniture", "Storage & Organization", "Vacuums", "Pools, Hot Tubs & Saunas", "Smart Appliances", "Home Safety & Security", "Trash & Recycling", "Skin Care", "Makeup", "Hair Care", "Bath & Body", "Oral Care", "Shaving & Hair Removal", "Vitamins & Nutrition", "Health & Wellness", "Fragrance", "Feminine Care", "Body Wash", "Moisturizers", "Personal Care", "Hair Care & Grooming", "Men's Personal Care & Grooming", "Home Spa & Massage", "Toys", "Outdoor Play", "Games & Puzzles", "Dress-up & Pretend Play", "Kids' Arts & Crafts", "Snacks", "Beverages", "Baking & Cooking Ingredients", "Canned & Packaged Food", "Sweets & Treats", "Breakfast Food & Beverages", "Condiments & Sauces", "Dairy & Eggs", "Fresh Flowers & Food Gift Baskets", "Vegan & Vegetarian Food", "Baked Goods", "Deli & Bakery", "Meat & Seafood", "Produce", "Cleaning Supplies", "Paper & Plastic Essentials", "Laundry Care", "Light Bulbs", "Insect & Pest Control", "Air Fresheners", "Hand Soap", "Fabric Freshening Sprays", "Hand Sanitizer", "Household Batteries & Accessories", "Moisture Absorbers", "Tools", "Heating, Cooling & Ventilation", "Light Fixtures & Ceiling Fans", "Plumbing Supplies", "Bathroom Home Improvement", "Building Supplies, Fasteners & Hardware", "Smart Home & Automation", "Electrical Supplies & Alternative Energy", "Doors & Windows", "Kitchen Home Improvement", "Material Handling Equipment", "Flooring & Carpet", "Women's Apparel", "Men's Apparel", "Kids' Apparel", "Shoes", "Jewelry", "Activewear", "Handbags & Accessories", "Baby & Toddler Apparel", "Wedding & Bridal", "Exercise & Fitness", "Athletics & Team Sports", "Cycling", "Camping & Hiking", "Swim, Boating & Water Sports", "Hunting & Fishing", "Golf", "Skates, Skateboards & Scooters", "Camping & Outdoors", "Fitness Trackers", "Water Bottles", "Snow & Winter Sports", "Kids' Sports & Active Play", "Game Room & Arcade", "Equestrian", "Outdoor & Lawn Games", "Tactical Gear", "ATV & UTV", "Paintball & Airsoft", "Party & Special Occasion", "Crafts & Hobbies", "Art Supplies", "Baby", "Kids", "Travel Electronics", "Bags, Packs & Totes", "Luggage", "Coolers & Cooler Bags", "Kids' Travel", "Travel Security", "Travel Accessories", "Dry Boxes", "Shopping Trolleys", "Dog Supplies", "Cat Supplies", "General Pet Supplies", "Fish & Aquarium Supplies", "Small Animal Supplies", "Reptile Supplies", "Pet Bird Supplies", "Books", "Movies & TV", "Music", "Magazines & Newspapers", "Vehicle Parts & Repair", "Vehicle Care & Maintenance", "Vehicle Accessories", "Vehicle Safety & Security", "Vehicle Storage & Cargo", "Motorcycle & Scooter", "Truck & Towing", "RV & Camper", "Vehicles", "Office Supplies", "Office Furniture", "Office Storage & Organization", "School Supplies", "Signage", "Commercial Appliances", "Science & Medical", "Business & Retail", "Restaurant & Food Service", "Commercial Safety & Security", "Commercial Facilities & Maintenance", "Farming & Agriculture", "Workplace Apparel & Uniforms", "Studio Recording", "Guitars & Accessories", "Live Sound, Stage & Lighting", "Pianos, Keyboards & Organs", "Musical Instrument Accessories", "Musical Instruction & Rehearsal", "DJ Equipment", "Drums & Percussion", "Jukeboxes & Karaoke Machines", "Folk & World Musical Instruments", "Wind & Woodwind Instruments", "Band & Orchestra", "String Instruments", "Sound Synthesizers", "Brass Instruments", "", "gold", "diamond", "luxury hotel", "luxury house"]

def write_category_data(data, file_path):
    with open(file_path, 'w') as output_file:
        json.dump(data, output_file)

def read_category_data(file_path):
    with open(file_path) as category_data_file:
        category_data = json.load(category_data_file)

        return category_data

def create_category_data_file():
    file_path = "./category_data.txt"
    if not os.path.exists(file_path):
        category_dict = {}
        for count, value in enumerate(category_list):
            category_dict[count] = {"name": value, "checked": True}
        write_category_data(category_dict, file_path)
    else:
        category_dict = read_category_data(file_path)
    
    return category_dict
    