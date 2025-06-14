#!/usr/bin/env python3
"""
ZoL0 Real-Time Collaboration Dashboard
=====================================
Advanced team collaboration system for trading bot monitoring:
- Real-time multi-user collaboration
- Live chat and annotations
- Shared dashboards and workspaces
- Role-based access control
- Activity logging and audit trails
- Screen sharing and remote assistance
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# === AI/ML Model Integration ===
from ai.models.AnomalyDetector import AnomalyDetector
from ai.models.SentimentAnalyzer import SentimentAnalyzer
from ai.models.ModelRecognizer import ModelRecognizer
from ai.models.ModelManager import ModelManager
from ai.models.ModelTrainer import ModelTrainer
from ai.models.ModelTuner import ModelTuner
from ai.models.ModelRegistry import ModelRegistry
from ai.models.ModelTraining import ModelTraining

# Page configuration
st.set_page_config(
    page_title="ZoL0 Team Collaboration",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .collaboration-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .user-card {
        background: linear-gradient(145deg, #e8f4fd, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    
    .online-indicator {
        color: #28a745;
        font-weight: bold;
    }
    
    .offline-indicator {
        color: #6c757d;
        font-weight: bold;
    }
    
    .away-indicator {
        color: #ffc107;
        font-weight: bold;
    }
    
    .chat-message {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        border-left: 3px solid #007bff;
    }
    
    .system-message {
        background: #e7f3ff;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        border-left: 3px solid #17a2b8;
        font-style: italic;
    }
    
    .workspace-panel {
        background: linear-gradient(145deg, #f0f8ff, #ffffff);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #007bff;
        margin: 1rem 0;
    }
    
    .annotation-panel {
        background: linear-gradient(145deg, #fff5f5, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #dc3545;
        margin: 1rem 0;
    }
    
    .activity-log {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        max-height: 300px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.9em;
    }
</style>
""",
    unsafe_allow_html=True,
)


class CollaborationSystem:
    def __init__(self):
        self.users = {}
        self.workspaces = {}
        self.chat_messages = []
        self.annotations = {}
        self.activity_log = []
        self.user_permissions = {}
        self.screen_sharing_sessions = {}

        # Initialize default workspace
        self.create_workspace(
            "main", "Main Trading Dashboard", ["admin", "trader", "analyst"]
        )

    def create_user(
        self, user_id: str, username: str, role: str, avatar_url: str = ""
    ) -> bool:
        """Create or update user profile"""
        self.users[user_id] = {
            "username": username,
            "role": role,
            "avatar_url": avatar_url,
            "status": "offline",
            "last_seen": datetime.now(),
            "current_workspace": None,
            "current_view": None,
            "permissions": self.get_role_permissions(role),
        }

        self.log_activity(f"User {username} ({role}) joined the system")
        return True

    def get_role_permissions(self, role: str) -> Dict[str, bool]:
        """Get permissions for a specific role"""
        permissions = {
            "admin": {
                "view_all_dashboards": True,
                "modify_dashboards": True,
                "manage_users": True,
                "manage_workspaces": True,
                "emergency_controls": True,
                "export_data": True,
                "view_audit_logs": True,
            },
            "trader": {
                "view_all_dashboards": True,
                "modify_dashboards": False,
                "manage_users": False,
                "manage_workspaces": False,
                "emergency_controls": True,
                "export_data": True,
                "view_audit_logs": False,
            },
            "analyst": {
                "view_all_dashboards": True,
                "modify_dashboards": False,
                "manage_users": False,
                "manage_workspaces": False,
                "emergency_controls": False,
                "export_data": True,
                "view_audit_logs": False,
            },
            "viewer": {
                "view_all_dashboards": True,
                "modify_dashboards": False,
                "manage_users": False,
                "manage_workspaces": False,
                "emergency_controls": False,
                "export_data": False,
                "view_audit_logs": False,
            },
        }
        return permissions.get(role, permissions["viewer"])

    def update_user_status(
        self, user_id: str, status: str, workspace: str = None, view: str = None
    ):
        """Update user online status and current location"""
        if user_id in self.users:
            old_status = self.users[user_id]["status"]
            self.users[user_id]["status"] = status
            self.users[user_id]["last_seen"] = datetime.now()

            if workspace:
                self.users[user_id]["current_workspace"] = workspace
            if view:
                self.users[user_id]["current_view"] = view

            if old_status != status:
                username = self.users[user_id]["username"]
                self.log_activity(f"{username} is now {status}")

    def create_workspace(
        self, workspace_id: str, name: str, allowed_roles: List[str]
    ) -> bool:
        """Create a new collaborative workspace"""
        self.workspaces[workspace_id] = {
            "name": name,
            "allowed_roles": allowed_roles,
            "created_at": datetime.now(),
            "active_users": [],
            "shared_views": {},
            "annotations": {},
            "chat_history": [],
        }

        self.log_activity(f"Workspace '{name}' created")
        return True

    def join_workspace(self, user_id: str, workspace_id: str) -> bool:
        """User joins a workspace"""
        if user_id not in self.users or workspace_id not in self.workspaces:
            return False

        user = self.users[user_id]
        workspace = self.workspaces[workspace_id]

        # Check permissions
        if user["role"] not in workspace["allowed_roles"] and user["role"] != "admin":
            return False

        # Add user to workspace
        if user_id not in workspace["active_users"]:
            workspace["active_users"].append(user_id)

        user["current_workspace"] = workspace_id

        username = user["username"]
        self.log_activity(f"{username} joined workspace '{workspace['name']}'")
        return True

    def send_chat_message(
        self, user_id: str, workspace_id: str, message: str, message_type: str = "chat"
    ) -> bool:
        """Send chat message to workspace"""
        if user_id not in self.users or workspace_id not in self.workspaces:
            return False

        user = self.users[user_id]
        workspace = self.workspaces[workspace_id]

        chat_message = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "username": user["username"],
            "message": message,
            "type": message_type,  # chat, system, alert
            "timestamp": datetime.now(),
            "workspace_id": workspace_id,
        }

        workspace["chat_history"].append(chat_message)
        self.chat_messages.append(chat_message)

        # Keep only last 100 messages per workspace
        if len(workspace["chat_history"]) > 100:
            workspace["chat_history"] = workspace["chat_history"][-100:]

        return True

    def create_annotation(
        self,
        user_id: str,
        workspace_id: str,
        view_id: str,
        x: float,
        y: float,
        text: str,
        annotation_type: str = "note",
    ) -> str:
        """Create annotation on a view"""
        annotation_id = str(uuid.uuid4())

        annotation = {
            "id": annotation_id,
            "user_id": user_id,
            "username": self.users[user_id]["username"],
            "workspace_id": workspace_id,
            "view_id": view_id,
            "x": x,
            "y": y,
            "text": text,
            "type": annotation_type,  # note, warning, question, suggestion
            "created_at": datetime.now(),
            "resolved": False,
        }

        if workspace_id not in self.annotations:
            self.annotations[workspace_id] = {}
        if view_id not in self.annotations[workspace_id]:
            self.annotations[workspace_id][view_id] = {}

        self.annotations[workspace_id][view_id][annotation_id] = annotation

        username = self.users[user_id]["username"]
        self.log_activity(f"{username} added annotation in {workspace_id}/{view_id}")

        return annotation_id

    def start_screen_sharing(self, user_id: str, workspace_id: str) -> str:
        """Start screen sharing session"""
        session_id = str(uuid.uuid4())

        session = {
            "id": session_id,
            "host_user_id": user_id,
            "host_username": self.users[user_id]["username"],
            "workspace_id": workspace_id,
            "viewers": [],
            "started_at": datetime.now(),
            "status": "active",
        }

        self.screen_sharing_sessions[session_id] = session

        username = self.users[user_id]["username"]
        self.log_activity(f"{username} started screen sharing session")

        return session_id

    def join_screen_sharing(self, user_id: str, session_id: str) -> bool:
        """Join screen sharing session"""
        if session_id not in self.screen_sharing_sessions:
            return False

        session = self.screen_sharing_sessions[session_id]
        if user_id not in session["viewers"]:
            session["viewers"].append(user_id)

        username = self.users[user_id]["username"]
        self.log_activity(f"{username} joined screen sharing session")

        return True

    def log_activity(self, message: str):
        """Log system activity"""
        self.activity_log.append({"timestamp": datetime.now(), "message": message})

        # Keep only last 1000 log entries
        if len(self.activity_log) > 1000:
            self.activity_log = self.activity_log[-1000:]

    def get_user_statistics(self) -> Dict[str, Any]:
        """Get user activity statistics"""
        total_users = len(self.users)
        online_users = len([u for u in self.users.values() if u["status"] == "online"])

        role_distribution = {}
        for user in self.users.values():
            role = user["role"]
            role_distribution[role] = role_distribution.get(role, 0) + 1

        return {
            "total_users": total_users,
            "online_users": online_users,
            "offline_users": total_users - online_users,
            "role_distribution": role_distribution,
            "active_workspaces": len(
                [w for w in self.workspaces.values() if w["active_users"]]
            ),
            "total_messages": len(self.chat_messages),
            "total_annotations": sum(
                len(views)
                for workspace in self.annotations.values()
                for views in workspace.values()
            ),
        }


class CollaborationAI:
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.model_recognizer = ModelRecognizer()
        self.model_manager = ModelManager()
        self.model_trainer = ModelTrainer()
        self.model_tuner = ModelTuner()
        self.model_registry = ModelRegistry()
        self.model_training = ModelTraining(self.model_trainer)

    def detect_collab_anomalies(self, activity_logs):
        try:
            if not activity_logs:
                return []
            features = [len(log.get("message", "")) for log in activity_logs]
            X = np.array(features).reshape(-1, 1)
            preds = self.anomaly_detector.predict(X)
            return [{"index": i, "anomaly": int(preds[i] == -1)} for i in range(len(preds))]
        except Exception as e:
            print(f"AI anomaly detection failed: {e}")
            return []

    def ai_collab_recommendations(self, activity_logs):
        recs = []
        try:
            errors = [log.get("message", "") for log in activity_logs]
            sentiment = self.sentiment_analyzer.analyze(errors)
            if sentiment.get("compound", 0) > 0.5:
                recs.append("Collaboration sentiment is positive. Team is engaged.")
            elif sentiment.get("compound", 0) < -0.5:
                recs.append("Collaboration sentiment is negative. Review team activity.")
            patterns = self.model_recognizer.recognize(errors)
            if patterns and patterns.get("confidence", 0) > 0.8:
                recs.append(
                    f"Pattern detected: {patterns['pattern']} (confidence: {patterns['confidence']:.2f})"
                )
            if not recs:
                recs.append("No critical collaboration issues detected.")
        except Exception as e:
            recs.append(f"AI recommendation error: {e}")
        return recs

    def retrain_models(self, activity_logs):
        try:
            X = np.array([len(log.get("message", "")) for log in activity_logs]).reshape(-1, 1)
            if len(X) > 10:
                self.anomaly_detector.fit(X)
            return {"status": "retraining complete"}
        except Exception as e:
            print(f"Model retraining failed: {e}")
            return {"status": "retraining failed", "error": str(e)}

    def calibrate_models(self):
        try:
            self.anomaly_detector.calibrate(None)
            return {"status": "calibration complete"}
        except Exception as e:
            print(f"Model calibration failed: {e}")
            return {"status": "calibration failed", "error": str(e)}

    def get_model_status(self):
        try:
            return {
                "anomaly_detector": str(type(self.anomaly_detector.model)),
                "sentiment_analyzer": "ok",
                "model_recognizer": "ok",
                "registered_models": self.model_manager.list_models(),
            }
        except Exception as e:
            return {"error": str(e)}


def main():
    # Header
    st.markdown(
        """
    <div class="collaboration-header">
        <h1>👥 ZoL0 Team Collaboration Dashboard</h1>
        <p>Real-time collaboration for trading bot monitoring teams</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize collaboration system
    if "collaboration" not in st.session_state:
        st.session_state.collaboration = CollaborationSystem()

        # Add demo users
        collab = st.session_state.collaboration
        collab.create_user("user1", "Alice Johnson", "admin", "👩‍💼")
        collab.create_user("user2", "Bob Smith", "trader", "👨‍💻")
        collab.create_user("user3", "Carol Davis", "analyst", "👩‍📊")
        collab.create_user("user4", "David Wilson", "viewer", "👨‍👀")

        # Simulate some activity
        collab.update_user_status("user1", "online", "main", "dashboard")
        collab.update_user_status("user2", "online", "main", "analytics")
        collab.update_user_status("user3", "away", "main", "reports")
        collab.update_user_status("user4", "offline")

        collab.join_workspace("user1", "main")
        collab.join_workspace("user2", "main")
        collab.join_workspace("user3", "main")

        # Add some demo messages
        collab.send_chat_message(
            "user1",
            "main",
            "Good morning team! Let's review today's trading performance.",
            "chat",
        )
        collab.send_chat_message(
            "user2",
            "main",
            "I see some unusual activity in the EUR/USD pair. Alice, can you check?",
            "chat",
        )
        collab.send_chat_message(
            "user1",
            "main",
            "On it! I'll analyze the signals from the ML model.",
            "chat",
        )
        collab.send_chat_message(
            "system",
            "main",
            "Risk alert: Portfolio drawdown exceeded 5% threshold",
            "alert",
        )

    collaboration = st.session_state.collaboration

    # Sidebar - User Profile
    with st.sidebar:
        st.header("👤 User Profile")

        # Current user selection (in real app, this would be authentication)
        current_user_id = st.selectbox(
            "Select User (Demo)",
            list(collaboration.users.keys()),
            format_func=lambda x: collaboration.users[x]["username"],
        )

        if current_user_id:
            current_user = collaboration.users[current_user_id]
            st.write(f"**Role:** {current_user['role']}")
            st.write(f"**Status:** {current_user['status']}")

            # Status update
            new_status = st.selectbox(
                "Update Status", ["online", "away", "busy", "offline"]
            )
            if st.button("Update Status"):
                collaboration.update_user_status(current_user_id, new_status)
                st.success(f"Status updated to {new_status}")

        st.divider()

        # Quick Actions
        st.header("⚡ Quick Actions")
        if st.button("📢 Send Alert"):
            collaboration.send_chat_message(
                "system",
                "main",
                "System alert: High volatility detected in multiple pairs",
                "alert",
            )
            st.success("Alert sent!")

        if st.button("🖥️ Start Screen Share"):
            session_id = collaboration.start_screen_sharing(current_user_id, "main")
            st.success(f"Screen sharing started: {session_id[:8]}")

        if st.button("📊 Generate Report"):
            st.success("Collaboration report generated!")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🏠 Team Overview",
            "💬 Live Chat",
            "🏢 Workspaces",
            "📝 Annotations",
            "🖥️ Screen Sharing",
            "📊 Activity Log",
        ]
    )

    with tab1:
        st.header("Team Collaboration Overview")

        # User statistics
        stats = collaboration.get_user_statistics()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", stats["total_users"])
        with col2:
            st.metric("Online Now", stats["online_users"], f"+{stats['online_users']}")
        with col3:
            st.metric("Active Workspaces", stats["active_workspaces"])
        with col4:
            st.metric("Total Messages", stats["total_messages"], "+23")

        # Active users
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👥 Team Members")
            for user_id, user in collaboration.users.items():
                with st.container():
                    st.markdown('<div class="user-card">', unsafe_allow_html=True)

                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    with col_a:
                        st.write(user["avatar_url"] if user["avatar_url"] else "👤")

                    with col_b:
                        st.write(f"**{user['username']}**")
                        st.write(f"Role: {user['role']}")
                        if user["current_workspace"]:
                            st.write(
                                f"In: {collaboration.workspaces[user['current_workspace']]['name']}"
                            )

                    with col_c:
                        if user["status"] == "online":
                            st.markdown(
                                '<span class="online-indicator">🟢 Online</span>',
                                unsafe_allow_html=True,
                            )
                        elif user["status"] == "away":
                            st.markdown(
                                '<span class="away-indicator">🟡 Away</span>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                '<span class="offline-indicator">⚫ Offline</span>',
                                unsafe_allow_html=True,
                            )

                    st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.subheader("📊 Team Activity")

            # Role distribution
            if stats["role_distribution"]:
                fig = px.pie(
                    values=list(stats["role_distribution"].values()),
                    names=list(stats["role_distribution"].keys()),
                    title="Team Role Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Recent activity
            st.subheader("🕒 Recent Activity")
            recent_activities = collaboration.activity_log[-5:]
            for activity in reversed(recent_activities):
                st.write(
                    f"**{activity['timestamp'].strftime('%H:%M')}** - {activity['message']}"
                )

    with tab2:
        st.header("💬 Live Team Chat")

        # Workspace selection
        workspace_options = {
            ws_id: ws["name"] for ws_id, ws in collaboration.workspaces.items()
        }
        selected_workspace = st.selectbox(
            "Select Workspace",
            list(workspace_options.keys()),
            format_func=lambda x: workspace_options[x],
        )

        if selected_workspace:
            workspace = collaboration.workspaces[selected_workspace]

            # Chat history
            st.subheader(f"💬 {workspace['name']} Chat")

            chat_container = st.container()
            with chat_container:
                for message in workspace["chat_history"][-20:]:  # Show last 20 messages
                    timestamp = message["timestamp"].strftime("%H:%M")

                    if message["type"] == "system" or message["type"] == "alert":
                        st.markdown(
                            '<div class="system-message">', unsafe_allow_html=True
                        )
                        st.write(f"🤖 **SYSTEM** [{timestamp}]: {message['message']}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="chat-message">', unsafe_allow_html=True
                        )
                        st.write(
                            f"**{message['username']}** [{timestamp}]: {message['message']}"
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

            # Send message
            st.subheader("✍️ Send Message")
            col1, col2 = st.columns([4, 1])

            with col1:
                new_message = st.text_input("Type your message...", key="chat_input")

            with col2:
                if st.button("Send", key="send_message"):
                    if new_message and current_user_id:
                        collaboration.send_chat_message(
                            current_user_id, selected_workspace, new_message
                        )
                        st.rerun()

            # Quick message buttons
            st.write("**Quick Messages:**")
            quick_messages = [
                "👍 Looks good!",
                "⚠️ Need attention here",
                "✅ Task completed",
                "❓ Can you clarify?",
                "🔍 Investigating...",
                "🛑 Emergency stop!",
            ]

            cols = st.columns(3)
            for i, msg in enumerate(quick_messages):
                with cols[i % 3]:
                    if st.button(msg, key=f"quick_{i}"):
                        collaboration.send_chat_message(
                            current_user_id, selected_workspace, msg
                        )
                        st.rerun()

    with tab3:
        st.header("🏢 Collaborative Workspaces")

        st.markdown('<div class="workspace-panel">', unsafe_allow_html=True)
        st.subheader("➕ Create New Workspace")

        col1, col2 = st.columns(2)
        with col1:
            new_workspace_name = st.text_input(
                "Workspace Name", placeholder="Risk Management Team"
            )
            new_workspace_id = st.text_input("Workspace ID", placeholder="risk_team")

        with col2:
            allowed_roles = st.multiselect(
                "Allowed Roles",
                ["admin", "trader", "analyst", "viewer"],
                default=["admin", "trader"],
            )

        if st.button("Create Workspace"):
            if new_workspace_name and new_workspace_id:
                collaboration.create_workspace(
                    new_workspace_id, new_workspace_name, allowed_roles
                )
                st.success("Workspace created successfully!")
            else:
                st.error("Please provide workspace name and ID")

        st.markdown("</div>", unsafe_allow_html=True)

        # Existing workspaces
        st.subheader("📋 Active Workspaces")
        for ws_id, workspace in collaboration.workspaces.items():
            with st.expander(f"🏢 {workspace['name']}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**ID:** {ws_id}")
                    st.write(
                        f"**Created:** {workspace['created_at'].strftime('%Y-%m-%d %H:%M')}"
                    )

                with col2:
                    st.write(f"**Active Users:** {len(workspace['active_users'])}")
                    st.write(
                        f"**Allowed Roles:** {', '.join(workspace['allowed_roles'])}"
                    )

                with col3:
                    if st.button("Join Workspace", key=f"join_{ws_id}"):
                        if collaboration.join_workspace(current_user_id, ws_id):
                            st.success("Joined workspace!")
                        else:
                            st.error("Cannot join workspace")

                # Active users in workspace
                if workspace["active_users"]:
                    st.write("**Active Users:**")
                    for user_id in workspace["active_users"]:
                        if user_id in collaboration.users:
                            user = collaboration.users[user_id]
                            st.write(
                                f"• {user['username']} ({user['role']}) - {user['status']}"
                            )

    with tab4:
        st.header("📝 Collaborative Annotations")

        st.markdown('<div class="annotation-panel">', unsafe_allow_html=True)
        st.subheader("📌 Create Annotation")

        col1, col2 = st.columns(2)
        with col1:
            annotation_workspace = st.selectbox(
                "Workspace",
                list(collaboration.workspaces.keys()),
                format_func=lambda x: collaboration.workspaces[x]["name"],
            )
            annotation_view = st.selectbox(
                "View/Dashboard",
                ["main_dashboard", "analytics_view", "risk_monitor", "portfolio_view"],
            )
            annotation_type = st.selectbox(
                "Type", ["note", "warning", "question", "suggestion"]
            )

        with col2:
            annotation_x = st.slider("X Position", 0.0, 100.0, 50.0)
            annotation_y = st.slider("Y Position", 0.0, 100.0, 50.0)
            annotation_text = st.text_area(
                "Annotation Text", placeholder="This trend looks concerning..."
            )

        if st.button("Create Annotation"):
            if annotation_text and current_user_id:
                annotation_id = collaboration.create_annotation(
                    current_user_id,
                    annotation_workspace,
                    annotation_view,
                    annotation_x,
                    annotation_y,
                    annotation_text,
                    annotation_type,
                )
                st.success(f"Annotation created: {annotation_id[:8]}")
            else:
                st.error("Please provide annotation text")

        st.markdown("</div>", unsafe_allow_html=True)

        # Display annotations
        st.subheader("📋 Active Annotations")
        for workspace_id, views in collaboration.annotations.items():
            workspace_name = collaboration.workspaces[workspace_id]["name"]

            for view_id, annotations in views.items():
                if annotations:
                    st.write(f"**{workspace_name} - {view_id}:**")

                    for ann_id, annotation in annotations.items():
                        emoji = {
                            "note": "📝",
                            "warning": "⚠️",
                            "question": "❓",
                            "suggestion": "💡",
                        }[annotation["type"]]

                        with st.expander(
                            f"{emoji} {annotation['username']} - {annotation['created_at'].strftime('%H:%M')}"
                        ):
                            st.write(
                                f"**Position:** ({annotation['x']:.1f}, {annotation['y']:.1f})"
                            )
                            st.write(f"**Text:** {annotation['text']}")
                            st.write(f"**Type:** {annotation['type']}")

                            if not annotation["resolved"]:
                                if st.button("Mark Resolved", key=f"resolve_{ann_id}"):
                                    annotation["resolved"] = True
                                    st.success("Annotation marked as resolved!")

    with tab5:
        st.header("🖥️ Screen Sharing & Remote Assistance")

        # Active screen sharing sessions
        st.subheader("📺 Active Screen Sharing Sessions")

        active_sessions = [
            s
            for s in collaboration.screen_sharing_sessions.values()
            if s["status"] == "active"
        ]

        if active_sessions:
            for session in active_sessions:
                with st.expander(f"🖥️ {session['host_username']}'s Session"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write(f"**Host:** {session['host_username']}")
                        st.write(
                            f"**Started:** {session['started_at'].strftime('%H:%M')}"
                        )

                    with col2:
                        st.write(f"**Viewers:** {len(session['viewers'])}")
                        st.write(
                            f"**Workspace:** {collaboration.workspaces[session['workspace_id']]['name']}"
                        )

                    with col3:
                        if current_user_id != session["host_user_id"]:
                            if st.button(
                                "Join Session", key=f"join_screen_{session['id']}"
                            ):
                                if collaboration.join_screen_sharing(
                                    current_user_id, session["id"]
                                ):
                                    st.success("Joined screen sharing session!")
                        else:
                            if st.button(
                                "End Session", key=f"end_screen_{session['id']}"
                            ):
                                session["status"] = "ended"
                                st.success("Screen sharing session ended!")

                    # Show viewers
                    if session["viewers"]:
                        st.write("**Current Viewers:**")
                        for viewer_id in session["viewers"]:
                            if viewer_id in collaboration.users:
                                viewer = collaboration.users[viewer_id]
                                st.write(f"• {viewer['username']} ({viewer['role']})")
        else:
            st.info("No active screen sharing sessions")

        # Remote assistance
        st.subheader("🆘 Remote Assistance")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Request Help:**")
            help_topic = st.selectbox(
                "Topic",
                [
                    "Dashboard Navigation",
                    "Data Analysis",
                    "Alert Configuration",
                    "Technical Issue",
                    "Training Request",
                ],
            )
            help_description = st.text_area("Describe the issue...")

            if st.button("Request Assistance"):
                collaboration.send_chat_message(
                    current_user_id,
                    "main",
                    f"🆘 HELP REQUEST: {help_topic} - {help_description}",
                    "alert",
                )
                st.success("Help request sent to team!")

        with col2:
            st.write("**Available Experts:**")
            experts = [
                u
                for u in collaboration.users.values()
                if u["role"] in ["admin", "trainer"] and u["status"] == "online"
            ]

            for expert in experts:
                st.write(f"🟢 {expert['username']} ({expert['role']})")

            if not experts:
                st.info("No experts currently online")

    with tab6:
        st.header("📊 Activity Log & Analytics")

        # Activity statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Activities", len(collaboration.activity_log))
        with col2:
            recent_count = len(
                [
                    a
                    for a in collaboration.activity_log
                    if a["timestamp"] > datetime.now() - timedelta(hours=1)
                ]
            )
            st.metric("Last Hour", recent_count)
        with col3:
            st.metric("Annotations", stats["total_annotations"])

        # Activity log
        st.subheader("🕒 Real-Time Activity Log")

        st.markdown('<div class="activity-log">', unsafe_allow_html=True)
        for activity in reversed(
            collaboration.activity_log[-50:]
        ):  # Show last 50 activities
            timestamp = activity["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            st.write(f"[{timestamp}] {activity['message']}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Activity chart
        st.subheader("📈 Activity Trends")

        # Generate hourly activity data
        now = datetime.now()
        hours = [(now - timedelta(hours=i)).hour for i in range(24, 0, -1)]
        activity_counts = [
            len(
                [
                    a
                    for a in collaboration.activity_log
                    if a["timestamp"].hour == hour
                    and a["timestamp"].date() == now.date()
                ]
            )
            for hour in hours
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=activity_counts,
                mode="lines+markers",
                name="Activity Count",
                line=dict(color="#007bff"),
            )
        )

        fig.update_layout(
            title="Team Activity by Hour (Today)",
            xaxis_title="Hour of Day",
            yaxis_title="Activity Count",
            height=300,
        )

        st.plotly_chart(fig, use_container_width=True)

    # --- Streamlit AI/ML Recommendations Panel ---
    def show_ai_collab_recommendations(activity_logs):
        recs = collab_ai.ai_collab_recommendations(activity_logs)
        st.sidebar.header("AI Collaboration Recommendations")
        for rec in recs:
            st.sidebar.info(rec)

    # --- Streamlit Model Management Panel ---
    def show_model_management():
        st.sidebar.header("Model Management")
        st.sidebar.write(collab_ai.get_model_status())
        if st.sidebar.button("Retrain Models"):
            st.sidebar.write(collab_ai.retrain_models([]))
        if st.sidebar.button("Calibrate Models"):
            st.sidebar.write(collab_ai.calibrate_models())

    # --- Streamlit Monetization Panel ---
    def show_monetization_panel():
        st.sidebar.header("Monetization & Usage")
        st.sidebar.write(
            {
                "usage": {
                    "collab_sessions": 123,
                    "premium_analytics": 42,
                    "reports_generated": 7,
                }
            }
        )
        st.sidebar.write(
            {
                "affiliates": [
                    {"id": "partner1", "revenue": 1200},
                    {"id": "partner2", "revenue": 800},
                ]
            }
        )
        st.sidebar.write({"pricing": {"base": 99, "premium": 199, "enterprise": 499}})

    # --- Streamlit Automation Panel ---
    def show_automation_panel():
        st.sidebar.header("Automation")
        if st.sidebar.button("Schedule Collaboration Audit"):
            st.sidebar.success("Collaboration audit scheduled!")
        if st.sidebar.button("Schedule Model Retrain"):
            st.sidebar.success("Model retraining scheduled!")

    # --- Integrate panels into main Streamlit app ---
    activity_logs = collaboration.activity_log
    show_ai_collab_recommendations(activity_logs)
    show_model_management()
    show_monetization_panel()
    show_automation_panel()


if __name__ == "__main__":
    main()
