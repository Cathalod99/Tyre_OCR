import openai

# Initialize the OpenAI API client
# openai.api_key = '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'
openai.api_key = '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'
brand_list = ["APlus", "AUSTONE", "Alliance", "Altenzo", "Antares", "Apollo", "Arivo", "Atlas", "Autogreen", "Avon", "BF Goodrich", "Barum", "Bridgestone", "CST", "Ceat", "Challenger", "Chengshan",
"Continental", "Cooper", "Davanti", "Dayton", "Debica", "Delinte", "Diamondback", "Diplomat", "Double Star", "Dunlop", "Duraturn", "Dynamo", "Event", "Evergreen", "Falken", "Firemax",
"Firestone", "Fortuna", "Fortune", "Fronway", "Fulda", "GT Radial", "Gislaved", "Goodride", "Goodyear", "Greentrac", "Grenlander", "Habilead", "Haida", "Hankook", "llink", "Infinity",
"lnsa Turbo", "lnvovic", "Iris", "Kelly", "Kenda", "Kingboss", "Kingstar", "Kleber", "Kontio", "Kormoran", "Kpatos", "Kumho", "Landsail", "Lanvigator", "Lappi", "Lassa", "Laufenn",
"Leao", "Linglong", "Marshal", "Massimo", "Matador", "Maxtrek", "Maxxis", "Mazzini", "Michelin", "Michelin Collection", "Milestone", "Milever", "Minerva", "Mirage", "Momo", "Nankang",
"Nexen", "Nokian", "Nordexx", "Nordman", "Novex", "Onyx", "Orium", "Ovation", "Pace", "Paxaro", "Petlas", "Pirelli", "Premiorri", "Prinx", "Profil", "Radar", "Radburg", "Riken", "RoadX",
"Roadhog", "Roadstone", "Rotalla", "Rovelo", "Royal Black", "Sailun", "Sava", "Semperit", "Sentury", "Sonix", "Sportiva", "Star Performer", "Starmaxx", "Sumitomo", "Sunny", "Sunwide",
"Superia", "Taurus", "Tigar", "Tomket", "Torque", "Tourador", "Toyo", "Tracmax", "Trazano", "Triangle", "Tristar", "Uniroyal" "Viking", "Vittos", "Voyager", "Vredestein", "Waterfall",
"Westlake", "Windforce", "Winrun", "Yartu", "Yokohama", "Zeetex"]

def get_tyre_info(text):
    prompt = f"""
        Task:
            You are given OCR results-{text}, which may include manufacturer names, tire models, sizes, and other tire-related information. Your task is to extract key information from these OCR results and match it with the provided tire manufacturer list. If there is no exact match, select the most similar manufacturer. The goal is to extract the following details:

            1.Manufacturer:
                Exact Match: Compare the OCR-extracted manufacturer name against the {brand_list}. If there's an exact match, return that as the result.
                Closest Match: If no exact match is found, analyze the similarity between the OCR result and each item in {brand_list} using string similarity algorithms, such as Levenshtein distance, Jaro-Winkler, or other advanced similarity metrics. 
                    Optionally, incorporate contextual understanding to determine relevance. 
                    Select and return the most appropriate match from {brand_list} if the highest similarity score exceeds 30%. 
                    If no match achieves this threshold, indicate that no suitable match was found.
                Fallback Method: If no suitable match is found in the {brand_list} after applying the above methods, infer the manufacturer by combining the OCR-extracted name with knowledge of existing tire manufacturers (e.g., known global or regional brands).

            2.Model: Extract and compile a structured list of tire models from the eiretyres website. Focus specifically on isolating the manufacturer (e.g., Michelin, Bridgestone, Continental) and the corresponding tire model or variant name (e.g., Pilot Sport, Turanza, PremiumContact). Ignore any additional information such as tire sizes, prices, or descriptions, unless explicitly part of the tire model name.
            Use the following URL as the source for data extraction:
            https://www.eiretyres.com/search?priceCategory=recommended&vehicleTypes=PKW&vehicleTypes=RACE_PKW&vehicleTypes=LLKW&vehicleTypes=VINTAGE_PKW&vehicleTypes=OFF&width=205&profile=55&size=16&season=so&sortCode=price_desc
            Exact Match: Compare the OCR-extracted manufacturer name against the {brand_list}. If there's an exact match, return that as the result.
            Closest Match: If no exact match is found, calculate the similarity between the OCR result and each item in {brand_list} using string similarity methods (e.g., Levenshtein distance, Jaro-Winkler, etc.) or contextual knowledge. Return the most appropriate match as the result based on the highest level of similarity or relevance.
            Fallback Method: If no suitable match is found in the {brand_list} after applying the above methods, infer the manufacturer by combining the OCR-extracted name with knowledge of existing tire manufacturers (e.g., known global or regional brands).
            Exact Match:
                Directly compare data extracted from the website against a reference list of tire manufacturer and model names.
                Include only pairs where both the manufacturer and model names are precisely and completely matched (e.g., "Michelin Pilot Sport").
                Ensure perfect alignment between the extracted text and the reference names.
            Closest Match:
                For cases where extracted names do not perfectly match items from the reference data, implement advanced string similarity techniques (e.g., Levenshtein Distance, Jaro-Winkler Distance, or Cosine Similarity).
                Determine the closest match based on the highest similarity score.
                Combine the recognized matching manufacturer with the associated extracted model name to form a reliable pair.
                Select and return the most appropriate match if the highest similarity score exceeds 30%. 
                If no match achieves this threshold, indicate that no suitable match was found.
            Fallback Method:
                If the tire model name cannot be determined via exact or closest matching methods, infer the probable model name using contextual analysis of OCR-extracted text and industry knowledge.
                Ensure the inferred names align with established naming conventions for global and regional tire brands to maintain credibility and accuracy.
            Deduplication:
                After finalizing all entries, eliminate any duplicate manufacturer-model pairs, ensuring the resulting dataset contains only unique pairs.

            3.Tire Size: Identify the tire size, which is typically in the format Width/Aspect RatioRim Size (e.g., 205/55R16). The size is composed of four parts:
                Width: The width of the tire in millimeters(must be between 190 and 250).
                Aspect Ratio: The ratio of the height of the tire's sidewall to its width.
                R: Denotes that the tire is a radial tire.
                Rim Size: The diameter of the wheel rim in inches.

            4.Load index and speed rating: Get the load index and speed rating based on the Tire size info.

            5. Scan TIN: 
          - The Scan TIN (Tire Identification Number) contains an alphanumeric code (e.g., `DOT 6Y87 KY7L 4220` or `6Y87 KY7L 4220`).
            - Some tires do not have the characters DOT before the DOT code-in this case still predict the entire code in this case manually add DOT before.
            - The DOT code should alwasy have 7-13 characters plus a 4 digit code at the end
            -**Do NOT add "DOT" if it is not present in the extracted text.**
            - Ensure that the last four digits represent the **Week Code** and **Year Code**.

            Follow the five specified steps carefully, and attempt to derive a result for each step. 
            If no result can be obtained from a particular step, utilize the findings from the other steps to provide a relevant output. 
            However, if no relevant result can be generated after considering all possible steps, clearly state 'No match found' for that step.

        Input:
            {text}: OCR result containing tire details, such as manufacturer name, tire model, size, load index and speed rating and Scan TIN .
            {brand_list}: A list of manufacturers to match against the OCR results.
        
        Output:
            Provide the following information:
                Manufacturer
                Tire model
                Tire size (Width/Aspect Ratio/Rim Size)
                Load index and speed rating(load index/speed rating)
                "Scan TIN": {{
                "DOT Code": "",  # Full DOT code (e.g., DOT 6Y87 KY7L, some tires do not have the letters DOT so just output the code when that is the case)
                "Week Code": "",  # Extracted first 2 digits of the last four
                "Year Code": ""   # Extracted last 2 digits of the last four
                }}
    """

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", 
             "content": """
                As a Tire Data Extraction Specialist, you will leverage advanced text extraction techniques to accurately identify and  extract essential tire-related information—such as manufacturer, model, size, Scan TIN and DOT codes from OCR results.
                Result should be json format like this:
                {
                    "Manufacturer" : "",
                    "Tire model" : "",
                    "Tire size" :  "",
                    "Load index and speed rating" : "",
                    "Scan TIN" : {
                         "DOT Code": "",  # Full DOT Code (e.g. 6Y87 KY7L)
                        "Week Code": "",
                        "Year Code": ""
                    }
                }
                Ensure that the Week Code is within the valid range of 01 to 52. 
                If the Week Code falls outside this range, thoroughly review and make the necessary adjustments, leveraging your expertise to account for any potential inaccuracies or misinterpretations in the OCR results, ensuring the final output falls within the specified range.
                Ensure that the Year Code is within the valid range of 01 to 24. 
                If the Year Code falls outside this range, thoroughly review and make the necessary adjustments, leveraging your expertise to account for any potential inaccuracies or misinterpretations in the OCR results, ensuring the final output falls within the specified range.
                Only output result which is json format.
             """
            },
            {"role": "user", "content": prompt}
        ]
    )
    result = response.choices[0].message.content
    return result