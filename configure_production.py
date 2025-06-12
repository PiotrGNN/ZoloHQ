#!/usr/bin/env python3
"""
ZoL0 Trading System - Production Configuration
Production environment configuration and setup utilities.
"""

import json
import logging
import os
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("configure_production.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""

    host: str = "localhost"
    port: int = 5432
    database: str = "zol0_trading"
    username: str = "zol0_user"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 10
    echo: bool = False
    backup_enabled: bool = True
    backup_interval_hours: int = 6


@dataclass
class SecurityConfig:
    """Security configuration"""

    api_key: str = ""
    jwt_secret: str = ""
    encryption_key: str = ""
    rate_limit_per_minute: int = 100
    enable_https: bool = True
    cors_origins: List[str] = None
    session_timeout_minutes: int = 60
    max_login_attempts: int = 5


@dataclass
class TradingConfig:
    """Trading configuration"""

    environment: str = "production"  # production, staging, demo
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_percentage: float = 0.02  # 2%
    take_profit_percentage: float = 0.05  # 5%
    max_daily_trades: int = 50
    enable_live_trading: bool = False
    paper_trading_balance: float = 100000.0
    risk_management_enabled: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""

    enable_system_monitoring: bool = True
    enable_trading_alerts: bool = True
    enable_performance_tracking: bool = True
    alert_email_recipients: List[str] = None
    webhook_urls: List[str] = None
    log_level: str = "INFO"
    log_retention_days: int = 30
    metrics_retention_days: int = 90


@dataclass
class APIConfig:
    """API server configuration"""

    host: str = "0.0.0.0"
    port: int = 5001
    workers: int = 4
    timeout: int = 30
    max_request_size: int = 16777216  # 16MB
    enable_swagger: bool = False
    enable_metrics: bool = True


class ProductionConfigurator:
    """Production environment configuration manager"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        # Configuration files
        self.main_config_file = self.config_dir / "production.json"
        self.secrets_file = self.config_dir / "secrets.json"
        self.env_file = self.config_dir / ".env"

        # Default configurations
        self.database_config = DatabaseConfig()
        self.security_config = SecurityConfig()
        self.trading_config = TradingConfig()
        self.monitoring_config = MonitoringConfig()
        self.api_config = APIConfig()

        # Load existing configuration
        self._load_configuration()

    def _load_configuration(self):
        """Load existing configuration files"""
        try:
            if self.main_config_file.exists():
                with open(self.main_config_file, "r") as f:
                    config_data = json.load(f)

                    # Load database config
                    if "database" in config_data:
                        db_data = config_data["database"]
                        self.database_config = DatabaseConfig(**db_data)

                    # Load trading config
                    if "trading" in config_data:
                        trading_data = config_data["trading"]
                        self.trading_config = TradingConfig(**trading_data)

                    # Load monitoring config
                    if "monitoring" in config_data:
                        monitoring_data = config_data["monitoring"]
                        self.monitoring_config = MonitoringConfig(**monitoring_data)

                    # Load API config
                    if "api" in config_data:
                        api_data = config_data["api"]
                        self.api_config = APIConfig(**api_data)

            # Load secrets
            if self.secrets_file.exists():
                with open(self.secrets_file, "r") as f:
                    secrets_data = json.load(f)
                    if "security" in secrets_data:
                        security_data = secrets_data["security"]
                        self.security_config = SecurityConfig(**security_data)

            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    def generate_secrets(self):
        """Generate secure secrets for production"""
        logger.info("Generating production secrets...")

        # Generate API key
        self.security_config.api_key = secrets.token_urlsafe(32)

        # Generate JWT secret
        self.security_config.jwt_secret = secrets.token_urlsafe(64)

        # Generate encryption key
        self.security_config.encryption_key = secrets.token_urlsafe(32)

        logger.info("Production secrets generated successfully")

    def configure_database(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        username: str = None,
        password: str = None,
    ):
        """Configure database settings"""
        if host:
            self.database_config.host = host
        if port:
            self.database_config.port = port
        if database:
            self.database_config.database = database
        if username:
            self.database_config.username = username
        if password:
            self.database_config.password = password

        logger.info("Database configuration updated")

    def configure_trading(
        self,
        environment: str = None,
        max_position_size: float = None,
        enable_live_trading: bool = None,
        paper_trading_balance: float = None,
    ):
        """Configure trading settings"""
        if environment:
            self.trading_config.environment = environment
        if max_position_size is not None:
            self.trading_config.max_position_size = max_position_size
        if enable_live_trading is not None:
            self.trading_config.enable_live_trading = enable_live_trading
        if paper_trading_balance is not None:
            self.trading_config.paper_trading_balance = paper_trading_balance

        logger.info("Trading configuration updated")

    def setup_directory_structure(self):
        """Create production directory structure"""
        logger.info("Setting up production directory structure...")

        directories = [
            "logs",
            "data",
            "cache",
            "backups",
            "config",
            "static",
            "templates",
            "uploads",
            "exports",
        ]

        for directory in directories:
            dir_path = Path(directory)
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")

        # Create subdirectories
        (Path("logs") / "trading").mkdir(exist_ok=True)
        (Path("logs") / "api").mkdir(exist_ok=True)
        (Path("logs") / "monitoring").mkdir(exist_ok=True)

        (Path("data") / "market_data").mkdir(exist_ok=True)
        (Path("data") / "portfolio").mkdir(exist_ok=True)
        (Path("data") / "analytics").mkdir(exist_ok=True)

        (Path("backups") / "database").mkdir(exist_ok=True)
        (Path("backups") / "config").mkdir(exist_ok=True)

        logger.info("Directory structure created successfully")

    def setup_logging_config(self):
        """Setup production logging configuration"""
        logger.info("Setting up logging configuration...")

        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
                },
                "simple": {"format": "%(asctime)s - %(levelname)s - %(message)s"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": "logs/zol0_trading.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "encoding": "utf8",
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "detailed",
                    "filename": "logs/errors.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 3,
                    "encoding": "utf8",
                },
                "trading_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "detailed",
                    "filename": "logs/trading/trading_activity.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 10,
                    "encoding": "utf8",
                },
            },
            "loggers": {
                "": {  # root logger
                    "level": "DEBUG",
                    "handlers": ["console", "file", "error_file"],
                },
                "trading": {
                    "level": "INFO",
                    "handlers": ["trading_file"],
                    "propagate": False,
                },
                "uvicorn": {"level": "INFO", "handlers": ["console"]},
            },
        }

        # Save logging configuration
        logging_config_file = self.config_dir / "logging.json"
        with open(logging_config_file, "w") as f:
            json.dump(logging_config, f, indent=2)

        logger.info("Logging configuration saved")

    def setup_systemd_services(self):
        """Setup systemd service files for production deployment"""
        logger.info("Setting up systemd service files...")

        # API server service
        api_service = f"""[Unit]
Description=ZoL0 Trading System API Server
After=network.target
Wants=network.target

[Service]
Type=simple
User=zol0
Group=zol0
WorkingDirectory={Path.cwd()}
Environment=PATH={Path.cwd()}/venv/bin
ExecStart={Path.cwd()}/venv/bin/python enhanced_dashboard_api.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=zol0-api

[Install]
WantedBy=multi-user.target
"""

        # Dashboard service
        dashboard_service = f"""[Unit]
Description=ZoL0 Trading System Dashboard
After=network.target
Wants=network.target

[Service]
Type=simple
User=zol0
Group=zol0
WorkingDirectory={Path.cwd()}
Environment=PATH={Path.cwd()}/venv/bin
ExecStart={Path.cwd()}/venv/bin/streamlit run unified_trading_dashboard.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=zol0-dashboard

[Install]
WantedBy=multi-user.target
"""

        # Bot monitor service
        monitor_service = f"""[Unit]
Description=ZoL0 Trading System Bot Monitor
After=network.target
Wants=network.target

[Service]
Type=simple
User=zol0
Group=zol0
WorkingDirectory={Path.cwd()}
Environment=PATH={Path.cwd()}/venv/bin
ExecStart={Path.cwd()}/venv/bin/python enhanced_bot_monitor.py --monitor
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=zol0-monitor

[Install]
WantedBy=multi-user.target
"""

        # Save service files
        services_dir = self.config_dir / "systemd"
        services_dir.mkdir(exist_ok=True)

        (services_dir / "zol0-api.service").write_text(api_service)
        (services_dir / "zol0-dashboard.service").write_text(dashboard_service)
        (services_dir / "zol0-monitor.service").write_text(monitor_service)

        logger.info("Systemd service files created in config/systemd/")
        logger.info("To install services, run as root:")
        logger.info("  cp config/systemd/*.service /etc/systemd/system/")
        logger.info("  systemctl daemon-reload")
        logger.info("  systemctl enable zol0-api zol0-dashboard zol0-monitor")
        logger.info("  systemctl start zol0-api zol0-dashboard zol0-monitor")

    def setup_nginx_config(self):
        """Setup nginx configuration for reverse proxy"""
        logger.info("Setting up nginx configuration...")

        nginx_config = """
# ZoL0 Trading System - Nginx Configuration
upstream zol0_api {
    server 127.0.0.1:5001;
}

upstream zol0_dashboard {
    server 127.0.0.1:8501;
}

# API Server
server {
    listen 80;
    server_name api.zol0trading.local;
    
    location / {
        proxy_pass http://zol0_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}

# Dashboard
server {
    listen 80;
    server_name dashboard.zol0trading.local;
    
    location / {
        proxy_pass http://zol0_dashboard;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    location /_stcore/stream {
        proxy_pass http://zol0_dashboard/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}

# Main Application (choose one)
server {
    listen 80 default_server;
    server_name zol0trading.local _;
    
    # Redirect to dashboard by default
    location / {
        return 301 http://dashboard.zol0trading.local$request_uri;
    }
    
    # API endpoints
    location /api/ {
        proxy_pass http://zol0_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
"""

        # Save nginx configuration
        nginx_dir = self.config_dir / "nginx"
        nginx_dir.mkdir(exist_ok=True)
        (nginx_dir / "zol0-trading.conf").write_text(nginx_config)

        logger.info("Nginx configuration saved to config/nginx/zol0-trading.conf")
        logger.info("To install nginx config:")
        logger.info(
            "  sudo cp config/nginx/zol0-trading.conf /etc/nginx/sites-available/"
        )
        logger.info(
            "  sudo ln -s /etc/nginx/sites-available/zol0-trading.conf /etc/nginx/sites-enabled/"
        )
        logger.info("  sudo nginx -t && sudo systemctl reload nginx")

    def create_environment_file(self):
        """Create .env file with environment variables"""
        logger.info("Creating environment file...")

        env_content = f"""# ZoL0 Trading System - Production Environment Variables

# Database Configuration
DATABASE_HOST={self.database_config.host}
DATABASE_PORT={self.database_config.port}
DATABASE_NAME={self.database_config.database}
DATABASE_USER={self.database_config.username}
DATABASE_PASSWORD={self.database_config.password}

# Security Configuration
API_KEY={self.security_config.api_key}
JWT_SECRET={self.security_config.jwt_secret}
ENCRYPTION_KEY={self.security_config.encryption_key}

# Trading Configuration
TRADING_ENVIRONMENT={self.trading_config.environment}
ENABLE_LIVE_TRADING={self.trading_config.enable_live_trading}
MAX_POSITION_SIZE={self.trading_config.max_position_size}
PAPER_TRADING_BALANCE={self.trading_config.paper_trading_balance}

# API Configuration
API_HOST={self.api_config.host}
API_PORT={self.api_config.port}
API_WORKERS={self.api_config.workers}

# Monitoring Configuration
LOG_LEVEL={self.monitoring_config.log_level}
ENABLE_SYSTEM_MONITORING={self.monitoring_config.enable_system_monitoring}

# Application Configuration
PYTHONPATH={Path.cwd()}
CONFIG_DIR={self.config_dir}
"""

        with open(self.env_file, "w") as f:
            f.write(env_content)

        # Set appropriate permissions
        os.chmod(self.env_file, 0o600)

        logger.info("Environment file created successfully")

    def setup_database_initialization(self):
        """Setup database initialization scripts"""
        logger.info("Setting up database initialization...")

        # Database initialization SQL
        init_sql = """
-- ZoL0 Trading System Database Initialization

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Trading accounts table
CREATE TABLE IF NOT EXISTS trading_accounts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    account_name VARCHAR(100) NOT NULL,
    broker VARCHAR(50) NOT NULL,
    account_type VARCHAR(20) DEFAULT 'demo',
    api_key_encrypted TEXT,
    api_secret_encrypted TEXT,
    balance DECIMAL(15,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Portfolio positions table
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES trading_accounts(id),
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(15,8) NOT NULL,
    avg_price DECIMAL(15,8) NOT NULL,
    current_price DECIMAL(15,8),
    unrealized_pnl DECIMAL(15,2),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading orders table
CREATE TABLE IF NOT EXISTS trading_orders (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES trading_accounts(id),
    symbol VARCHAR(20) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(15,8) NOT NULL,
    price DECIMAL(15,8),
    stop_price DECIMAL(15,8),
    status VARCHAR(20) DEFAULT 'pending',
    order_id_external VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filled_at TIMESTAMP,
    cancelled_at TIMESTAMP
);

-- Market data table
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open_price DECIMAL(15,8),
    high_price DECIMAL(15,8),
    low_price DECIMAL(15,8),
    close_price DECIMAL(15,8),
    volume DECIMAL(20,8),
    timeframe VARCHAR(10) NOT NULL
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES trading_accounts(id),
    date DATE NOT NULL,
    total_value DECIMAL(15,2),
    daily_pnl DECIMAL(15,2),
    cumulative_pnl DECIMAL(15,2),
    max_drawdown DECIMAL(15,2),
    sharpe_ratio DECIMAL(10,4),
    win_rate DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System alerts table
CREATE TABLE IF NOT EXISTS system_alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    data JSONB,
    is_read BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_account_symbol ON portfolio_positions(account_id, symbol);
CREATE INDEX IF NOT EXISTS idx_trading_orders_account_status ON trading_orders(account_id, status);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_account_date ON performance_metrics(account_id, date);
CREATE INDEX IF NOT EXISTS idx_system_alerts_created_at ON system_alerts(created_at);

-- Insert default admin user (password: 'admin123' - change in production!)
INSERT INTO users (username, email, password_hash, role) 
VALUES ('admin', 'admin@zol0trading.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiCw8.SVCG4.', 'admin')
ON CONFLICT (username) DO NOTHING;
"""

        # Save initialization script
        db_dir = self.config_dir / "database"
        db_dir.mkdir(exist_ok=True)
        (db_dir / "init.sql").write_text(init_sql)

        logger.info("Database initialization script created")

    def save_configuration(self):
        """Save all configuration to files"""
        logger.info("Saving configuration files...")

        # Main configuration
        main_config = {
            "database": asdict(self.database_config),
            "trading": asdict(self.trading_config),
            "monitoring": asdict(self.monitoring_config),
            "api": asdict(self.api_config),
        }

        with open(self.main_config_file, "w") as f:
            json.dump(main_config, f, indent=2)

        # Security configuration (secrets)
        security_config = {"security": asdict(self.security_config)}

        with open(self.secrets_file, "w") as f:
            json.dump(security_config, f, indent=2)

        # Set appropriate permissions for secrets
        os.chmod(self.secrets_file, 0o600)

        logger.info("Configuration files saved successfully")

    def create_backup_script(self):
        """Create backup script for production"""
        logger.info("Creating backup script...")

        backup_script = f"""#!/bin/bash
# ZoL0 Trading System - Backup Script

BACKUP_DIR="{Path.cwd()}/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATABASE_BACKUP_DIR="$BACKUP_DIR/database"
CONFIG_BACKUP_DIR="$BACKUP_DIR/config"

# Create backup directories
mkdir -p "$DATABASE_BACKUP_DIR"
mkdir -p "$CONFIG_BACKUP_DIR"

# Database backup
echo "Creating database backup..."
pg_dump -h {self.database_config.host} -p {self.database_config.port} -U {self.database_config.username} -d {self.database_config.database} > "$DATABASE_BACKUP_DIR/zol0_trading_$TIMESTAMP.sql"

# Configuration backup
echo "Creating configuration backup..."
tar -czf "$CONFIG_BACKUP_DIR/config_$TIMESTAMP.tar.gz" config/

# Data backup
echo "Creating data backup..."
tar -czf "$BACKUP_DIR/data_$TIMESTAMP.tar.gz" data/ --exclude="data/cache"

# Cleanup old backups (keep last 30 days)
find "$BACKUP_DIR" -name "*.sql" -mtime +30 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $TIMESTAMP"
"""

        backup_script_file = self.config_dir / "backup.sh"
        with open(backup_script_file, "w") as f:
            f.write(backup_script)

        # Make executable
        os.chmod(backup_script_file, 0o755)

        # Create crontab entry suggestion
        crontab_entry = f"""# ZoL0 Trading System Backup (runs every 6 hours)
0 */6 * * * {backup_script_file}
"""

        (self.config_dir / "crontab_backup").write_text(crontab_entry)

        logger.info("Backup script created successfully")
        logger.info(
            f"To install backup cron job: crontab -e and add: {crontab_entry.strip()}"
        )

    def setup_production_environment(self):
        """Complete production environment setup"""
        logger.info("Starting complete production environment setup...")

        try:
            # Generate secrets
            self.generate_secrets()

            # Setup directory structure
            self.setup_directory_structure()

            # Setup logging
            self.setup_logging_config()

            # Create environment file
            self.create_environment_file()

            # Setup database
            self.setup_database_initialization()

            # Setup system services
            self.setup_systemd_services()

            # Setup nginx
            self.setup_nginx_config()

            # Create backup script
            self.create_backup_script()

            # Save all configuration
            self.save_configuration()

            logger.info("Production environment setup completed successfully!")

            # Print summary
            self._print_setup_summary()

        except Exception as e:
            logger.error(f"Error during production setup: {e}")
            raise

    def _print_setup_summary(self):
        """Print setup summary and next steps"""
        print("\n" + "=" * 60)
        print("ZoL0 Trading System - Production Setup Complete")
        print("=" * 60)
        print(f"Configuration Directory: {self.config_dir}")
        print(f"Environment File: {self.env_file}")
        print(f"API Port: {self.api_config.port}")
        print("Dashboard Port: 8501")
        print(f"Trading Environment: {self.trading_config.environment}")
        print(f"Live Trading Enabled: {self.trading_config.enable_live_trading}")

        print("\nNext Steps:")
        print("1. Review and update configuration files in config/")
        print("2. Set up database server and run config/database/init.sql")
        print("3. Install systemd services (see logs for commands)")
        print("4. Setup nginx reverse proxy (see logs for commands)")
        print("5. Install SSL certificates for HTTPS")
        print("6. Setup backup cron job")
        print("7. Configure monitoring and alerting")
        print("8. Test all services")

        print("\nImportant Security Notes:")
        print("- Change default admin password immediately")
        print("- Secure database with proper credentials")
        print("- Enable firewall and restrict access")
        print("- Use HTTPS in production")
        print("- Regularly backup data and configuration")
        print("=" * 60)


def main():
    """Main function for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="ZoL0 Trading System - Production Configuration"
    )
    parser.add_argument(
        "--setup", "-s", action="store_true", help="Run complete production setup"
    )
    parser.add_argument(
        "--config-dir", "-c", default="config", help="Configuration directory"
    )
    parser.add_argument("--database-host", help="Database host")
    parser.add_argument("--database-port", type=int, help="Database port")
    parser.add_argument("--database-name", help="Database name")
    parser.add_argument("--database-user", help="Database username")
    parser.add_argument("--database-password", help="Database password")
    parser.add_argument(
        "--environment",
        choices=["production", "staging", "demo"],
        help="Trading environment",
    )
    parser.add_argument(
        "--enable-live-trading", action="store_true", help="Enable live trading"
    )

    args = parser.parse_args()

    configurator = ProductionConfigurator(args.config_dir)

    # Update configuration based on arguments
    if (
        args.database_host
        or args.database_port
        or args.database_name
        or args.database_user
        or args.database_password
    ):
        configurator.configure_database(
            host=args.database_host,
            port=args.database_port,
            database=args.database_name,
            username=args.database_user,
            password=args.database_password,
        )

    if args.environment or args.enable_live_trading:
        configurator.configure_trading(
            environment=args.environment, enable_live_trading=args.enable_live_trading
        )

    if args.setup:
        configurator.setup_production_environment()
    else:
        print("Use --setup to run complete production setup")
        print("Use --help for available options")


if __name__ == "__main__":
    main()
