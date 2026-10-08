const analysisData = JSON.parse(
    document.getElementById("analysis-data").textContent
);
document.addEventListener("DOMContentLoaded", function () {

    // Resume Score
    document.getElementById("resumeScore").textContent =
        analysisData.overall_score;

    // ATS Score
    document.getElementById("atsScore").textContent =
        analysisData.ats_score;

    // Skills Match
    let skillsMatch = 0;

    if (analysisData.skills && analysisData.skills.length > 0) {
        skillsMatch = Math.min(100, analysisData.skills.length * 15);
    }

    document.getElementById("skillsScore").textContent = skillsMatch;

    // Readability
    if (analysisData.ats_details) {
        document.getElementById("readabilityScore").textContent =
            analysisData.ats_details.readability;
    }

    // Resume Score Status
    document.getElementById("resumeStatus").textContent =
        analysisData.overall_score >= 70 ? "Good" :
            analysisData.overall_score >= 40 ? "Average" : "Needs Improvement";

    // ATS Status
    document.getElementById("atsStatus").textContent =
        analysisData.ats_score >= 70 ? "Good" :
            analysisData.ats_score >= 40 ? "Average" : "Needs Improvement";

    // Skills Status
    document.getElementById("skillsStatus").textContent =
        skillsMatch >= 70 ? "Good" :
            skillsMatch >= 40 ? "Average" : "Needs Improvement";

    // Readability Status
    document.getElementById("readabilityStatus").textContent =
        analysisData.ats_details.readability >= 70 ? "Good" :
            analysisData.ats_details.readability >= 40 ? "Average" : "Needs Improvement";


    // Strengths
    const strengthsList = document.getElementById("strengthsList");
    strengthsList.innerHTML = "";

    analysisData.strengths.forEach(function (strength) {
        const li = document.createElement("li");
        li.textContent = strength;
        strengthsList.appendChild(li);
    });


    // Areas for Improvement
    const improvementsList = document.getElementById("improvementsList");
    improvementsList.innerHTML = "";

    analysisData.weaknesses.forEach(function (weakness) {
        const li = document.createElement("li");
        li.textContent = weakness;
        improvementsList.appendChild(li);
    });


    // Missing Skills
    const missingSkills = document.getElementById("missingSkills");
    missingSkills.innerHTML = "";

    if (analysisData.ats_details.sections_missing.length > 0) {

        analysisData.ats_details.sections_missing.forEach(function (skill) {
            const span = document.createElement("span");
            span.textContent = skill;
            missingSkills.appendChild(span);
        });

    } else {
        const span = document.createElement("span");
        span.textContent = "No major missing sections";
        missingSkills.appendChild(span);
    }


    // Suggestions
    const suggestionsList = document.getElementById("suggestionsList");
    suggestionsList.innerHTML = "";

    analysisData.suggestions.forEach(function (suggestion) {
        const li = document.createElement("li");
        li.textContent = suggestion;
        suggestionsList.appendChild(li);
    });


    // Download button
    const downloadBtn = document.getElementById("downloadBtn");

    downloadBtn.addEventListener("click", function () {
        window.print();
    });


    // Preview button
    const previewBtn = document.getElementById("previewBtn");

    previewBtn.addEventListener("click", function () {
        window.open("/preview-resume", "_blank");
    });

});