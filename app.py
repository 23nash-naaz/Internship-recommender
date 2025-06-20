import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from datetime import datetime
import PyPDF2
import io

class InternshipRecommender:
    """
    A class that implements an internship recommendation system using
    natural language processing and similarity matching.
    """
    
    def __init__(self):
        """Initialize the recommendation system with sample internship data."""
        self.internships = self._load_internship_data()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._process_internship_data()
        
    def _load_internship_data(self):
        """Load sample internship data (in a real system, this would come from a database)."""
        internships = [
            {
                "id": 1,
                "title": "Machine Learning Research Intern",
                "company": "TechInnovate Inc.",
                "description": "Work on cutting-edge ML algorithms for computer vision applications.",
                "requirements": "Python, TensorFlow, PyTorch, Computer Vision, ML fundamentals",
                "location": "Remote",
                "duration": "3 months",
                "stipend": "$2000/month",
                "category": "Data Science"
            },
            {
                "id": 2,
                "title": "Web Development Intern",
                "company": "WebSolutions",
                "description": "Develop responsive web applications using modern frameworks.",
                "requirements": "HTML, CSS, JavaScript, React, Node.js",
                "location": "New York, NY",
                "duration": "6 months",
                "stipend": "$1800/month",
                "category": "Software Development"
            },
            {
                "id": 3,
                "title": "Data Science Intern",
                "company": "DataMinds",
                "description": "Analyze large datasets and build predictive models.",
                "requirements": "Python, SQL, Pandas, NumPy, Scikit-learn, Statistics",
                "location": "San Francisco, CA",
                "duration": "4 months",
                "stipend": "$2500/month",
                "category": "Data Science"
            },
            {
                "id": 4,
                "title": "Mobile App Development Intern",
                "company": "AppCrafters",
                "description": "Create innovative mobile applications for iOS and Android platforms.",
                "requirements": "Swift, Kotlin, Flutter, Mobile UI/UX design",
                "location": "Austin, TX",
                "duration": "3 months",
                "stipend": "$1900/month",
                "category": "Software Development"
            },
            {
                "id": 5,
                "title": "Natural Language Processing Intern",
                "company": "AI Solutions",
                "description": "Work on NLP projects including chatbots and sentiment analysis.",
                "requirements": "Python, NLTK, spaCy, Transformers, Deep Learning",
                "location": "Remote",
                "duration": "5 months",
                "stipend": "$2200/month",
                "category": "Data Science"
            },
            {
                "id": 6,
                "title": "Cybersecurity Research Intern",
                "company": "SecureNet",
                "description": "Research and implement security protocols and conduct penetration testing.",
                "requirements": "Network Security, Python, Cryptography, Linux",
                "location": "Boston, MA",
                "duration": "6 months",
                "stipend": "$2100/month",
                "category": "Cybersecurity"
            },
            {
                "id": 7,
                "title": "UX/UI Design Intern",
                "company": "DesignHub",
                "description": "Create user-friendly interfaces and improve user experience for web and mobile apps.",
                "requirements": "Figma, Adobe XD, UI/UX principles, Prototyping",
                "location": "Seattle, WA",
                "duration": "4 months",
                "stipend": "$1700/month",
                "category": "Design"
            },
            {
                "id": 8,
                "title": "Cloud Computing Intern",
                "company": "CloudTech",
                "description": "Work with cloud services to build scalable and resilient applications.",
                "requirements": "AWS, Azure, Docker, Kubernetes, CI/CD",
                "location": "Chicago, IL",
                "duration": "5 months",
                "stipend": "$2300/month",
                "category": "DevOps"
            },
            {
                "id": 9,
                "title": "Blockchain Development Intern",
                "company": "ChainInnovate",
                "description": "Develop blockchain solutions and smart contracts.",
                "requirements": "Solidity, Web3.js, Ethereum, Cryptography",
                "location": "Remote",
                "duration": "3 months",
                "stipend": "$2400/month",
                "category": "Blockchain"
            },
            {
                "id": 10,
                "title": "Robotics Engineering Intern",
                "company": "RoboWorks",
                "description": "Design and implement algorithms for autonomous robots.",
                "requirements": "ROS, Python, C++, Computer Vision, Control Systems",
                "location": "Detroit, MI",
                "duration": "6 months",
                "stipend": "$2600/month",
                "category": "Robotics"
            },
        ]
        return pd.DataFrame(internships)
    
    def _process_internship_data(self):
        """Process internship data to create a searchable index."""
        # Combine relevant fields for matching
        self.internships['combined_text'] = self.internships.apply(
            lambda row: ' '.join([
                str(row['title']),
                str(row['description']),
                str(row['requirements']),
                str(row['category'])
            ]), axis=1
        )
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.internships['combined_text'])
        
    def get_recommendations(self, student_profile, top_n=5):
        """
        Get internship recommendations based on student profile.
        
        Args:
            student_profile (dict): Student information including skills, interests, etc.
            top_n (int): Number of recommendations to return
            
        Returns:
            list: Top matching internship opportunities
        """
        # Extract relevant info from student profile
        profile_text = ' '.join([
            student_profile.get('skills', ''),
            student_profile.get('interests', ''),
            student_profile.get('experience', ''),
            student_profile.get('education', '')
        ])
        
        # Apply location filter if specified
        filtered_internships = self.internships
        if student_profile.get('preferred_location'):
            location_pref = student_profile['preferred_location'].lower()
            if location_pref != 'any':
                filtered_internships = self.internships[
                    self.internships['location'].str.lower().str.contains('remote') |
                    self.internships['location'].str.lower().str.contains(location_pref)
                ]
                
                if len(filtered_internships) == 0:
                    return {
                        "message": f"No internships found in {student_profile['preferred_location']}. Consider searching with 'Any' location.",
                        "recommendations": []
                    }
        
        # Transform student profile into the same vector space
        profile_vector = self.vectorizer.transform([profile_text])
        
        # Get the indices of filtered internships within the original dataframe
        if len(filtered_internships) < len(self.internships):
            # Create a new TF-IDF matrix for the filtered internships
            filtered_combined_text = filtered_internships['combined_text'].tolist()
            filtered_tfidf_matrix = self.vectorizer.transform(filtered_combined_text)
            
            # Compute similarity scores for filtered internships
            sim_scores = cosine_similarity(profile_vector, filtered_tfidf_matrix).flatten()
        else:
            # Use the pre-computed TF-IDF matrix for all internships
            sim_scores = cosine_similarity(profile_vector, self.tfidf_matrix).flatten()
        
        # Get top N internship indices
        top_indices = sim_scores.argsort()[-top_n:][::-1]
        
        # Get the actual internships and their similarity scores
        recommendations = []
        for idx in top_indices:
            internship = filtered_internships.iloc[idx].to_dict()
            internship['similarity_score'] = round(sim_scores[idx] * 100, 2)  # Convert to percentage
            
            # Remove the combined_text field from the output
            if 'combined_text' in internship:
                del internship['combined_text']
                
            recommendations.append(internship)
        
        # Add AI explanation
        explanation = self._generate_recommendation_explanation(student_profile, recommendations)
        
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "student_profile_summary": self._summarize_student_profile(student_profile),
            "recommendations": recommendations,
            "explanation": explanation
        }
    
    def _summarize_student_profile(self, profile):
        """Generate a summary of the student profile used for recommendations."""
        return {
            "skills_matched": re.findall(r'\b\w+\b', profile.get('skills', '').lower()),
            "interests_matched": re.findall(r'\b\w+\b', profile.get('interests', '').lower()),
            "education_level": profile.get('education', 'Not specified')
        }
    
    def _generate_recommendation_explanation(self, profile, recommendations):
        """Generate an explanation of why these recommendations were selected."""
        
        if not recommendations:
            return "No matches found based on your profile."
        
        # Extract key skills from top recommendation
        top_rec = recommendations[0]
        top_req = top_rec.get('requirements', '').lower()
        
        # Extract skills from profile
        profile_skills = profile.get('skills', '').lower()
        
        # Find matching skills
        profile_skill_list = re.findall(r'\b\w+\b', profile_skills)
        req_skill_list = re.findall(r'\b\w+\b', top_req)
        
        matching_skills = [skill for skill in profile_skill_list if any(
            skill in req_word for req_word in req_skill_list
        )]
        
        if not matching_skills:
            matching_skills = profile_skill_list[:3] if profile_skill_list else ["your skills"]
        
        # Generate explanation
        explanation = (
            f"Based on your profile, you have strong alignment with {top_rec['category']} internships. "
            f"Your skills in {', '.join(matching_skills[:3])} match well with the requirements "
            f"of several opportunities, particularly the {top_rec['title']} role at {top_rec['company']}."
        )
        
        return explanation

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text

def parse_resume(resume_text):
    """
    Parse resume text to extract key information.
    This is a very basic implementation and would need more sophisticated NLP in a real system.
    """
    # Simple keyword-based extraction
    skills_keywords = [
        "python", "java", "javascript", "html", "css", "react", "node", "sql", "tensorflow",
        "pytorch", "data analysis", "machine learning", "ai", "aws", "azure", "docker", 
        "kubernetes", "git", "agile", "scrum", "c++", "c#", "php", "ruby", "swift", "kotlin",
        "flutter", "mobile", "web", "front-end", "back-end", "full-stack", "design", "ui/ux",
        "figma", "adobe", "photoshop", "illustrator", "xd", "tableau", "power bi", "excel",
        "statistics", "calculus", "linear algebra", "algorithms", "data structures", "cybersecurity",
        "networking", "linux", "windows", "macos", "blockchain", "ethereum", "solidity", "crypto"
    ]
    
    interests_keywords = [
        "data science", "artificial intelligence", "web development", "mobile development",
        "cloud computing", "cybersecurity", "blockchain", "machine learning", "deep learning",
        "computer vision", "natural language processing", "robotics", "iot", "game development",
        "ar/vr", "ui/ux design", "database management", "devops", "research", "open source",
        "project management", "entrepreneurship", "fintech", "healthtech", "edtech"
    ]
    
    education_patterns = [
        r"(bachelor|master|phd|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?|b\.?e\.?|m\.?e\.?|b\.?tech|m\.?tech)",
        r"(computer science|cs|information technology|it|software engineering|electrical engineering|data science)",
        r"(freshman|sophomore|junior|senior|1st year|2nd year|3rd year|4th year|graduate)"
    ]
    
    # Extract skills
    skills = []
    for keyword in skills_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', resume_text.lower()):
            skills.append(keyword)
    
    # Extract interests
    interests = []
    for keyword in interests_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', resume_text.lower()):
            interests.append(keyword)
    
    # Simple education extraction
    education = "Not specified"
    for pattern in education_patterns:
        matches = re.findall(pattern, resume_text.lower())
        if matches:
            education = "Student in " + ", ".join(set(matches))
            break
    
    # Extract experience (very basic)
    experience_years = re.search(r'(\d+)\s*(\+)?\s*(year|yr)s?\s*(of)?\s*experience', resume_text.lower())
    if experience_years:
        experience = f"{experience_years.group(1)}{experience_years.group(2) or ''} years of experience"
    else:
        experience = "Experience information not detected"
    
    return {
        "skills": ", ".join(skills),
        "interests": ", ".join(interests),
        "education": education,
        "experience": experience
    }

def main():
    st.set_page_config(
        page_title="Internship Recommender",
        page_icon="🎓",
        layout="wide",
    )

    st.title("🎓 Internship Recommendation System")
    st.markdown("Find the perfect internship opportunity that matches your skills and interests!")

    # Initialize session state for the student profile
    if 'student_profile' not in st.session_state:
        st.session_state.student_profile = {
            "name": "",
            "education": "",
            "skills": "",
            "interests": "",
            "experience": "",
            "preferred_location": "Any"
        }
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Resume", "Fill Out Form"])

    with tab1:
        st.header("Upload Your Resume")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="resume_upload")
        
        if uploaded_file is not None:
            # Extract text from the PDF
            with st.spinner("Analyzing your resume..."):
                try:
                    resume_text = extract_text_from_pdf(uploaded_file)
                    profile_data = parse_resume(resume_text)
                    
                    # Update session state with extracted data
                    st.session_state.student_profile.update(profile_data)
                    
                    # Show the extracted information with editing capabilities
                    st.subheader("Extracted Information (Edit if needed)")
                    
                    # Allow user to edit the extracted information
                    st.session_state.student_profile["skills"] = st.text_area(
                        "Skills", 
                        st.session_state.student_profile["skills"],
                        help="Comma-separated list of your technical skills",
                        key="skills_pdf"
                    )
                    
                    st.session_state.student_profile["interests"] = st.text_area(
                        "Interests", 
                        st.session_state.student_profile["interests"],
                        help="Comma-separated list of your professional interests",
                        key="interests_pdf"
                    )
                    
                    st.session_state.student_profile["education"] = st.text_input(
                        "Education", 
                        st.session_state.student_profile["education"],
                        key="education_pdf"
                    )
                    
                    st.session_state.student_profile["experience"] = st.text_area(
                        "Experience", 
                        st.session_state.student_profile["experience"],
                        key="experience_pdf"
                    )
                    
                    locations = ["Any", "Remote", "New York, NY", "San Francisco, CA", 
                                "Austin, TX", "Boston, MA", "Seattle, WA", "Chicago, IL", "Detroit, MI"]
                    st.session_state.student_profile["preferred_location"] = st.selectbox(
                        "Preferred Location", 
                        options=locations,
                        index=0,
                        key="location_pdf"  # Add a unique key for this selectbox
                    )
                    
                    if st.button("Get Recommendations", key="rec_from_resume"):
                        recommender = InternshipRecommender()
                        st.session_state.recommendations = recommender.get_recommendations(st.session_state.student_profile)
                
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.info("Please try using the form instead.")

    with tab2:
        st.header("Enter Your Information")
        
        st.session_state.student_profile["name"] = st.text_input(
            "Name", 
            st.session_state.student_profile["name"],
            key="name_form"
        )
        
        st.session_state.student_profile["education"] = st.text_input(
            "Education (e.g., Computer Science undergraduate, 3rd year)", 
            st.session_state.student_profile["education"],
            key="education_form"
        )
        
        st.session_state.student_profile["skills"] = st.text_area(
            "Skills (comma-separated)", 
            st.session_state.student_profile["skills"],
            help="E.g., Python, Java, Data Analysis, Machine Learning",
            key="skills_form"
        )
        
        st.session_state.student_profile["interests"] = st.text_area(
            "Interests (comma-separated)", 
            st.session_state.student_profile["interests"],
            help="E.g., Artificial Intelligence, Web Development, Cybersecurity",
            key="interests_form"
        )
        
        st.session_state.student_profile["experience"] = st.text_area(
            "Relevant Experience", 
            st.session_state.student_profile["experience"],
            help="Brief description of your past experience (if any)",
            key="experience_form"
        )
        
        locations = ["Any", "Remote", "New York, NY", "San Francisco, CA", 
                    "Austin, TX", "Boston, MA", "Seattle, WA", "Chicago, IL", "Detroit, MI"]
        st.session_state.student_profile["preferred_location"] = st.selectbox(
            "Preferred Location", 
            options=locations,
            index=locations.index(st.session_state.student_profile["preferred_location"]) if st.session_state.student_profile["preferred_location"] in locations else 0,
            key="location_form"  # Add a unique key for this selectbox
        )
        
        if st.button("Get Recommendations", key="rec_from_form"):
            recommender = InternshipRecommender()
            st.session_state.recommendations = recommender.get_recommendations(st.session_state.student_profile)

    # Display recommendations if available
    if st.session_state.recommendations:
        st.header("📋 Your Personalized Internship Recommendations")
        
        # Display the AI explanation
        st.markdown(f"### AI Insight\n{st.session_state.recommendations['explanation']}")
        
        # Display the recommendations in cards
        st.subheader("Top Matches")
        
        # Create columns for displaying internships
        cols = st.columns(3)
        
        for i, rec in enumerate(st.session_state.recommendations['recommendations']):
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:15px;">
                        <h3 style="color:#1E90FF;">{rec['title']}</h3>
                        <h4>{rec['company']} • {rec['location']}</h4>
                        <p><strong>Match Score:</strong> {rec['similarity_score']}%</p>
                        <p><strong>Category:</strong> {rec['category']}</p>
                        <p><strong>Duration:</strong> {rec['duration']}</p>
                        <p><strong>Stipend:</strong> {rec['stipend']}</p>
                        
                        <details>
                            <summary>More details</summary>
                            <p><strong>Description:</strong> {rec['description']}</p>
                            <p><strong>Requirements:</strong> {rec['requirements']}</p>
                        </details>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Profile summary
        with st.expander("View Your Profile Summary"):
            st.subheader("Skills Matched")
            st.write(", ".join(st.session_state.recommendations['student_profile_summary']['skills_matched']))
            
            st.subheader("Interests Matched")
            st.write(", ".join(st.session_state.recommendations['student_profile_summary']['interests_matched']))
            
            st.subheader("Education Level")
            st.write(st.session_state.recommendations['student_profile_summary']['education_level'])
        
if __name__ == "__main__":
    main()