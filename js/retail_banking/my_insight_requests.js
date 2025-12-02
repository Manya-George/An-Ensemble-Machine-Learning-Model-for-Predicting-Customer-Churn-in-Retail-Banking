// my_prediction_requests.js
const API_URL = "http://localhost:5000/api";

document.addEventListener("DOMContentLoaded", () => {

    const myRequestsList = document.getElementById("myRequestsList");

    function showErrorBox(message) {
        myRequestsList.innerHTML = `
            <div style="padding:16px; background:#fee2e2; color:#b91c1c; border-radius:8px;">
                <strong>Error:</strong> ${message}
            </div>
        `;
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }

    async function loadMyRequests() {
        try {
            const response = await fetch(`${API_URL}/my-insights-requests`, {
                credentials: "include"
            });

            const raw = await response.text();
            console.log("RAW API RESPONSE:", raw);

            let json;
            try {
                json = JSON.parse(raw);
            } catch (err) {
                console.error("Failed to parse JSON:", err);
                showErrorBox(
                    "Server returned an unexpected response (HTML instead of JSON). " +
                    "This usually means you are not authenticated."
                );
                return;
            }

            if (!json.success) {
                showErrorBox(json.message || "Could not load your requests.");
                return;
            }

            renderRequests(json.requests || []);

        } catch (error) {
            console.error("Error loading requests:", error);
            showErrorBox("Connection error — unable to reach the server.");
        }
    }

    function renderRequests(list) {
        if (!list.length) {
            myRequestsList.innerHTML = `
                <div style="padding:16px; background:#fff; border-radius:8px;">
                    You have no prediction requests yet.
                </div>
            `;
            return;
        }

        const html = list
            .map((req) => {
                const statusBadge =
                    req.status === "pending"
                        ? `<span style="color:#1E88E5; font-weight:700;">PENDING</span>`
                        : `<span style="color:#16a34a; font-weight:700;">RESOLVED</span>`;

                return `
                <div style="
                    background:white;
                    padding:16px;
                    border-radius:8px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.08);
                    margin-bottom:12px;
                ">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="font-size:14px; font-weight:600; margin-bottom:6px; color:#374151;">
                                Request #${req.insightID} for jobID #${req.jobID} • ${new Date(req.requestedAt).toLocaleString()}
                            </div>

                            <div style="font-size:14px; margin-top:4px; color:#6b7280;">
                                ${escapeHtml(req.department)} — ${statusBadge}
                            </div>
                        </div>

                        <div style="text-align:right;">
                            ${
                                req.status === "resolved"
                                    ? `
                                        <div style="font-size:14px;">Job ID: <strong>${req.jobID}</strong></div>
                                        <div style="color:#6b7280; font-size:12px;">
                                            Resolved by ${escapeHtml(req.resolvedByUsername || "Business analyst")}
                                        </div>
                                    `
                                    : `<div style="color:#6b7280; font-size:12px;">Awaiting Business Analyst</div>`
                            }
                        </div>
                    </div>

                    <div style="margin-top:12px; color:#374151;">
                        ${escapeHtml(req.description)}
                    </div>
                </div>
            `;
            })
            .join("");

        myRequestsList.innerHTML = html;
    }

    // Load requests on page load
    loadMyRequests();
});
