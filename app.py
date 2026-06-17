"""
AI Job Matching System - FastAPI Backend
Uses Cohere's Embed + Rerank models for intelligent job matching
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import cohere
import json
import os

app = FastAPI(
    title="AI Job Matching API",
    description="Intelligent job matching powered by Cohere AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Cohere client
# Replace with your actual API key or set COHERE_API_KEY env variable
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "your-cohere-api-key-here")
co = cohere.Client(COHERE_API_KEY)

# ─── Sample Job Database ───────────────────────────────────────────────────────
JOBS = [
    {
        "id": 1,
        "title": "Senior Machine Learning Engineer",
        "company": "TechCorp AI",
        "location": "San Francisco, CA (Hybrid)",
        "salary": "$150,000 - $200,000",
        "type": "Full-time",
        "description": "Design and deploy ML models at scale. Work with TensorFlow, PyTorch, and cloud infrastructure. 5+ years experience required. Responsibilities include building recommendation systems, NLP pipelines, and computer vision models. Strong Python skills essential.",
        "skills": ["Python", "TensorFlow", "PyTorch", "MLOps", "AWS", "Docker", "Kubernetes"],
        "experience": "5+ years"
    },
    {
        "id": 2,
        "title": "Frontend React Developer",
        "company": "StartupXYZ",
        "location": "Remote",
        "salary": "$90,000 - $130,000",
        "type": "Full-time",
        "description": "Build beautiful, performant web applications using React and TypeScript. Collaborate with designers to implement pixel-perfect UIs. Experience with Next.js, GraphQL, and modern CSS frameworks required.",
        "skills": ["React", "TypeScript", "Next.js", "GraphQL", "Tailwind CSS", "Redux"],
        "experience": "3+ years"
    },
    {
        "id": 3,
        "title": "Data Scientist",
        "company": "Analytics Pro",
        "location": "New York, NY",
        "salary": "$110,000 - $150,000",
        "type": "Full-time",
        "description": "Extract insights from large datasets using statistical analysis and machine learning. Build predictive models and data pipelines. Work with SQL, Python, and BI tools to drive business decisions.",
        "skills": ["Python", "R", "SQL", "Pandas", "Scikit-learn", "Tableau", "Statistics"],
        "experience": "3+ years"
    },
    {
        "id": 4,
        "title": "DevOps / Cloud Engineer",
        "company": "CloudBase Inc",
        "location": "Austin, TX (Hybrid)",
        "salary": "$120,000 - $160,000",
        "type": "Full-time",
        "description": "Architect and manage cloud infrastructure on AWS and GCP. Implement CI/CD pipelines, Kubernetes clusters, and infrastructure as code with Terraform. Strong Linux background required.",
        "skills": ["AWS", "GCP", "Kubernetes", "Terraform", "CI/CD", "Linux", "Docker"],
        "experience": "4+ years"
    },
    {
        "id": 5,
        "title": "NLP Research Scientist",
        "company": "AI Research Lab",
        "location": "Seattle, WA",
        "salary": "$160,000 - $220,000",
        "type": "Full-time",
        "description": "Conduct cutting-edge research in natural language processing and large language models. Publish papers, implement transformers, fine-tune LLMs. PhD preferred. Work on RAG systems, embeddings, and generative AI.",
        "skills": ["NLP", "LLMs", "Transformers", "Python", "PyTorch", "Hugging Face", "Research"],
        "experience": "PhD + 2 years"
    },
    {
        "id": 6,
        "title": "Full Stack Engineer",
        "company": "Product Studio",
        "location": "Remote",
        "salary": "$100,000 - $140,000",
        "type": "Full-time",
        "description": "Build and maintain full-stack web applications using Node.js, React, and PostgreSQL. Design RESTful APIs, implement authentication, and work with cloud deployments. Strong communication skills required.",
        "skills": ["Node.js", "React", "PostgreSQL", "REST APIs", "AWS", "TypeScript"],
        "experience": "3+ years"
    },
    {
        "id": 7,
        "title": "Computer Vision Engineer",
        "company": "VisionTech",
        "location": "Boston, MA",
        "salary": "$130,000 - $175,000",
        "type": "Full-time",
        "description": "Develop real-time computer vision systems for autonomous vehicles and robotics. Experience with OpenCV, deep learning object detection models (YOLO, Detectron2), and edge deployment required.",
        "skills": ["Computer Vision", "OpenCV", "PyTorch", "YOLO", "C++", "Python", "CUDA"],
        "experience": "4+ years"
    },
    {
        "id": 8,
        "title": "Backend Python Developer",
        "company": "FinTech Solutions",
        "location": "Chicago, IL (Hybrid)",
        "salary": "$95,000 - $130,000",
        "type": "Full-time",
        "description": "Build scalable backend services using Python, Django/FastAPI, and microservices architecture. Work with financial data, implement secure APIs, and optimize database performance with PostgreSQL and Redis.",
        "skills": ["Python", "Django", "FastAPI", "PostgreSQL", "Redis", "Microservices", "Docker"],
        "experience": "3+ years"
    },
    {
        "id": 9,
        "title": "AI Product Manager",
        "company": "InnovateCo",
        "location": "San Francisco, CA",
        "salary": "$140,000 - $180,000",
        "type": "Full-time",
        "description": "Lead the product strategy for AI-powered products. Work closely with ML engineers, designers, and business stakeholders to define roadmaps. Understanding of ML concepts, user research, and agile methodologies required.",
        "skills": ["Product Management", "AI/ML", "Agile", "Data Analysis", "User Research", "Roadmapping"],
        "experience": "5+ years"
    },
    {
        "id": 10,
        "title": "Cybersecurity Analyst",
        "company": "SecureNet",
        "location": "Washington, DC",
        "salary": "$100,000 - $140,000",
        "type": "Full-time",
        "description": "Protect enterprise systems from cyber threats. Conduct penetration testing, vulnerability assessments, incident response, and SIEM management. CISSP or CEH certification preferred.",
        "skills": ["Cybersecurity", "Penetration Testing", "SIEM", "Network Security", "Python", "CISSP"],
        "experience": "4+ years"
    }
]

# Pre-build job documents for embedding
JOB_DOCUMENTS = [
    f"{job['title']} at {job['company']}. {job['description']} Skills: {', '.join(job['skills'])}."
    for job in JOBS
]


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class CandidateProfile(BaseModel):
    name: str
    summary: str
    skills: List[str]
    experience_years: Optional[int] = 0
    preferred_location: Optional[str] = ""
    education: Optional[str] = ""


class MatchRequest(BaseModel):
    candidate: CandidateProfile
    top_k: Optional[int] = 5


class JobMatch(BaseModel):
    job: dict
    score: float
    relevance_index: int


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "AI Job Matching API is running!", "jobs_count": len(JOBS)}


@app.get("/jobs")
def get_all_jobs():
    return {"jobs": JOBS, "total": len(JOBS)}


@app.post("/match")
async def match_jobs(request: MatchRequest):
    """
    Core endpoint: Takes a candidate profile and returns ranked job matches
    using Cohere's Rerank model for semantic relevance scoring.
    """
    try:
        # Build candidate query from profile
        candidate = request.candidate
        query = f"""
        Candidate: {candidate.name}
        Summary: {candidate.summary}
        Skills: {', '.join(candidate.skills)}
        Experience: {candidate.experience_years} years
        Location preference: {candidate.preferred_location or 'Flexible'}
        Education: {candidate.education or 'Not specified'}
        """

        # Use Cohere Rerank to score jobs against the candidate profile
        rerank_response = co.rerank(
            query=query,
            documents=JOB_DOCUMENTS,
            top_n=request.top_k,
            model="rerank-english-v3.0"
        )

        # Build response with ranked jobs
        matches = []
        for result in rerank_response.results:
            job = JOBS[result.index]
            matches.append({
                "job": job,
                "score": round(result.relevance_score * 100, 1),
                "relevance_index": result.index,
                "match_reasons": _generate_match_reasons(candidate, job)
            })

        return {
            "candidate": candidate.name,
            "total_jobs_analyzed": len(JOBS),
            "top_matches": matches,
            "model_used": "cohere/rerank-english-v3.0"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching error: {str(e)}")


@app.post("/embed-match")
async def embed_match_jobs(request: MatchRequest):
    """
    Alternative endpoint using Cohere Embed for semantic similarity matching.
    Embeds candidate profile and jobs, then computes cosine similarity.
    """
    try:
        candidate = request.candidate
        candidate_text = f"{candidate.summary} Skills: {', '.join(candidate.skills)}"

        # Embed candidate profile
        candidate_embedding = co.embed(
            texts=[candidate_text],
            model="embed-english-v3.0",
            input_type="search_query"
        ).embeddings[0]

        # Embed all jobs
        job_embeddings = co.embed(
            texts=JOB_DOCUMENTS,
            model="embed-english-v3.0",
            input_type="search_document"
        ).embeddings

        # Compute cosine similarities
        import numpy as np
        candidate_vec = np.array(candidate_embedding)
        scores = []
        for i, job_emb in enumerate(job_embeddings):
            job_vec = np.array(job_emb)
            cosine_sim = np.dot(candidate_vec, job_vec) / (
                np.linalg.norm(candidate_vec) * np.linalg.norm(job_vec)
            )
            scores.append((i, float(cosine_sim)))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = scores[:request.top_k]

        matches = []
        for idx, score in top_matches:
            job = JOBS[idx]
            matches.append({
                "job": job,
                "score": round((score + 1) / 2 * 100, 1),  # normalize to 0-100
                "relevance_index": idx,
                "match_reasons": _generate_match_reasons(candidate, job)
            })

        return {
            "candidate": candidate.name,
            "total_jobs_analyzed": len(JOBS),
            "top_matches": matches,
            "model_used": "cohere/embed-english-v3.0"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")


@app.get("/jobs/{job_id}")
def get_job(job_id: int):
    job = next((j for j in JOBS if j["id"] == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _generate_match_reasons(candidate: CandidateProfile, job: dict) -> List[str]:
    """Generate human-readable match reasons by comparing skills"""
    reasons = []
    candidate_skills_lower = [s.lower() for s in candidate.skills]
    job_skills_lower = [s.lower() for s in job["skills"]]

    matched_skills = [
        s for s in job["skills"]
        if s.lower() in candidate_skills_lower
    ]

    if matched_skills:
        reasons.append(f"Matched skills: {', '.join(matched_skills)}")

    if candidate.preferred_location and (
        candidate.preferred_location.lower() in job["location"].lower()
        or "remote" in job["location"].lower()
    ):
        reasons.append(f"Location match: {job['location']}")

    return reasons if reasons else ["Semantic profile match via Cohere AI"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
