import os
import sqlite3
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types

app = FastAPI(title="The MAYOR VETERINARY CONSORTIUM - Enterprise Engine")
DB_FILE = "veterinary_consortium.db"

# 1. Ensure our core relational database is configured
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vet_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            suspected_disease TEXT,
            animal_species TEXT,
            case_count INTEGER,
            location TEXT,
            interventions TEXT,
            preventive_recommendations TEXT,
            raw_input TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---- INPUT STRUCTURES ----
class VetCaseData(BaseModel):
    suspected_disease: str = Field(description="The primary suspected or confirmed animal disease.")
    animal_species: str = Field(description="Type of animal affected (e.g., Cattle, Goat, Sheep).")
    case_count: int = Field(description="Number of animals showing symptoms.")
    location: str = Field(description="Geographic area or farm location mentioned.")
    interventions_given: List[str] = Field(description="Medications or treatments administered.")
    preventive_recommendations: List[str] = Field(description="Vaccines or management changes recommended.")

class VetInput(BaseModel):
    raw_notes: str

class FarmerQueryInput(BaseModel):
    farmer_location: str
    animal_type: str

class CountyGovtQuery(BaseModel):
    target_county_or_location: str = Field(description="The specific region the government wants to analyze.")

class PharmaMarketQuery(BaseModel):
    target_region: str = Field(description="The region where they want to predict drug/vaccine market demand.")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ---- AUTOMATIC HOME REDIRECT ----
@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")

# ---- PORTAL 1: FIELD VETERINARIANS (Data Collection Engine) ----
@app.post("/api/vet/submit-case", tags=["Veterinarian Portal"])
def submit_vet_case(data: VetInput):
    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=f"Extract clinical data from these notes: {data.raw_notes}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VetCaseData,
                system_instruction="You are the core data extraction engine for The MAYOR VETERINARY CONSORTIUM. Motto: Precision on Prevention, diagnostic and curative interventions.",
            ),
        )
        parsed_data = json.loads(response.text)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO vet_cases (suspected_disease, animal_species, case_count, location, interventions, preventive_recommendations, raw_input, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            parsed_data.get("suspected_disease"), parsed_data.get("animal_species"), parsed_data.get("case_count"),
            parsed_data.get("location"), str(parsed_data.get("interventions_given")), str(parsed_data.get("preventive_recommendations")),
            data.raw_notes, current_time
        ))
        conn.commit()
        conn.close()
        return {"status": "Success", "message": "Case safely saved", "data": parsed_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- PORTAL 2: FARMER INTEL (Localized Advisory Service) ----
@app.post("/api/farmer/get-advisory", tags=["Farmer Portal"])
def get_farmer_advisory(data: FarmerQueryInput):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT suspected_disease, preventive_recommendations FROM vet_cases WHERE location LIKE ?", (f"%{data.farmer_location}%",))
        local_history = cursor.fetchall()
        conn.close()
        
        history_context = "Recent cases logged by field vets here:\n" + "\n".join([f"- Disease: {r[0]}, Tips: {r[1]}" for r in local_history]) if local_history else "No direct recent vet records found in this exact ward."
        current_month = datetime.now().strftime("%B")
        
        prompt = f"Location: {data.farmer_location}\nLivestock: {data.animal_type}\nMonth: {current_month}\n\n{history_context}\n\nProvide a scannable preventative roadmap highlighting seasonal variations, high-risk localized diseases, and crucial required vaccines."
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction="You are the lead preventative AI agricultural specialist for The MAYOR VETERINARY CONSORTIUM."),
        )
        return {"current_season": current_month, "advisory": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- PORTAL 3: COUNTY GOVERNMENTS (B2G Disease Surveillance Dashboard) ----
@app.post("/api/government/surveillance-report", tags=["Government B2G Portal"])
def get_government_surveillance(data: CountyGovtQuery):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT suspected_disease, animal_species, case_count, timestamp FROM vet_cases WHERE location LIKE ?", (f"%{data.target_county_or_location}%",))
        raw_cases = cursor.fetchall()
        conn.close()
        
        if not raw_cases:
            return {"region": data.target_county_or_location, "status": "Clear", "message": "No active disease trends or outbreaks detected in our database."}
            
        case_summary = "\n".join([f"- {row[2]} heads of {row[1]} showing signs of {row[0]} logged on {row[3]}" for row in raw_cases])
        
        prompt = f"""
        Analyze these raw field veterinary reports for the region of {data.target_county_or_location}:
        {case_summary}
        
        Generate a formal Epidemiological Intelligence Report for the County Director of Veterinary Services. Include:
        1. An Outbreak Risk Assessment (Low/Medium/High)
        2. Hotspot Mapping Analysis
        3. Strategic Public Interventions & Mandatory Quarantine/Vaccination Directives required.
        """
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction="You are a Chief Veterinary Epidemiologist for The MAYOR VETERINARY CONSORTIUM reporting directly to state and local governments."),
        )
        return {"target_region": data.target_county_or_location, "total_logged_incidents": len(raw_cases), "epidemiological_report": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- PORTAL 4: PHARMACEUTICAL COMPANIES & STOCKISTS (B2B Demand Forecasting) ----
@app.post("/api/pharmaceutical/market-demand", tags=["Pharmaceutical B2B Portal"])
def get_pharma_demand(data: PharmaMarketQuery):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT suspected_disease, interventions, preventive_recommendations FROM vet_cases WHERE location LIKE ?", (f"%{data.target_region}%",))
        market_data = cursor.fetchall()
        conn.close()
        
        current_month = datetime.now().strftime("%B")
        clinical_context = "\n".join([f"- Disease: {row[0]}, Current Drug Used: {row[1]}, Vaccine Needs: {row[2]}" for row in market_data]) if market_data else "No active clinical logs for this area."
        
        prompt = f"""
        Current Month: {current_month}
        Target Region: {data.target_region}
        Recent veterinary treatment logs:
        {clinical_context}
        
        Predict commercial pharmaceutical demands for medical stockists. Highlight high-demand vaccines, looming antibiotic requirements, and strategic supply-chain preparation directives based on real-time disease vectors.
        """
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction="You are an expert Pharmaceutical Supply Chain Analyst specializing in veterinary medicine market-intelligence for The MAYOR VETERINARY CONSORTIUM."),
        )
        return {"target_region": data.target_region, "analysis_month": current_month, "market_demand_forecast": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
