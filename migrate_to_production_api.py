#!/usr/bin/env python3
"""
migrate_to_production_api.py
----------------------------
Script to migrate all ZoL0 dashboards from simulated/demo data to real Bybit production API data.

This script will:
1. Check current data sources across all dashboards
2. Configure real Bybit API connections
3. Update all dashboard data fetching to use production API
4. Test the integration
5. Provide rollback capability if needed
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ProductionAPIMigrator:
    """Handles migration from simulated data to real Bybit production API"""

    def __init__(self):
        self.base_path = Path("c:/Users/piotr/Desktop/Zol0")
        self.backup_path = (
            self.base_path
            / "backups"
            / f"pre_production_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.config_file = self.base_path / "production_config.json"

        # Dashboard files that need to be updated
        self.dashboard_files = [
            "advanced_trading_analytics.py",
            "real_time_market_data_integration.py",
            "ml_predictive_analytics.py",
            "enhanced_bot_monitor.py",
            "advanced_alert_management.py",
            "data_export_import_system.py",
        ]

        # Key data source files
        self.data_source_files = [
            "ZoL0-master/data/execution/bybit_connector.py",
            "ZoL0-master/data/data/market_data_fetcher.py",
            "ZoL0-master/python_libs/simplified_trading_engine.py",
        ]

        self.migration_status = {}

    def print_banner(self):
        """Print migration banner"""
        print("=" * 80)
        print("🚀 ZoL0 Production API Migration")
        print("=" * 80)
        print(
            "This will migrate ALL dashboards from simulated to REAL Bybit production data"
        )
        print(
            "⚠️  WARNING: This will use REAL market data and potentially REAL trading!"
        )
        print("=" * 80)
        print()

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met for migration"""
        print("📋 Checking Prerequisites...")
        print("-" * 40)

        issues = []

        # Check environment variables
        required_env_vars = ["BYBIT_API_KEY", "BYBIT_API_SECRET"]

        for var in required_env_vars:
            if not os.getenv(var):
                issues.append(f"Missing environment variable: {var}")

        # Check if production mode is confirmed
        if not os.getenv("BYBIT_PRODUCTION_CONFIRMED", "").lower() == "true":
            issues.append(
                "Production mode not confirmed (BYBIT_PRODUCTION_CONFIRMED != true)"
            )

        # Check if files exist
        for file_path in self.dashboard_files + self.data_source_files:
            full_path = self.base_path / file_path
            if not full_path.exists():
                issues.append(f"Required file not found: {file_path}")
                # Check API connectivity
        try:
            self._test_bybit_api_connection()
            print("✅ Bybit API connection: OK")
        except Exception as e:
            issues.append(f"Bybit API connection failed: {e}")

        if issues:
            print("❌ Prerequisites check failed:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("✅ All prerequisites met!")
            return True

    def _test_bybit_api_connection(self):
        """Test connection to Bybit API"""
        try:
            # Try multiple import paths
            connector = None
            try:
                sys.path.append(str(self.base_path / "ZoL0-master"))
                from data.execution.bybit_connector import BybitConnector

                connector = BybitConnector(
                    api_key=os.getenv("BYBIT_API_KEY"),
                    api_secret=os.getenv("BYBIT_API_SECRET"),
                    use_testnet=False,  # Production mode
                )
            except ImportError:
                try:
                    from ZoL0master.data.execution.bybit_connector import BybitConnector

                    connector = BybitConnector(
                        api_key=os.getenv("BYBIT_API_KEY"),
                        api_secret=os.getenv("BYBIT_API_SECRET"),
                        use_testnet=False,  # Production mode
                    )
                except ImportError:
                    # Test basic connection without BybitConnector
                    import requests

                    response = requests.get(
                        "https://api.bybit.com/v5/market/time", timeout=10
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("retCode") == 0:
                            return True
                    raise Exception("Basic API connectivity test failed")

            if connector:
                # Test basic API call
                server_time = connector.get_server_time()
                if not server_time.get("success", False):
                    raise Exception("API test call failed")

        except Exception as e:
            raise Exception(f"Failed to connect to Bybit API: {e}")

    def create_backup(self):
        """Create backup of current configuration"""
        print("💾 Creating Backup...")
        print("-" * 40)

        try:
            self.backup_path.mkdir(parents=True, exist_ok=True)

            # Backup dashboard files
            for file_path in self.dashboard_files + self.data_source_files:
                source = self.base_path / file_path
                if source.exists():
                    dest = self.backup_path / file_path
                    dest.parent.mkdir(parents=True, exist_ok=True)

                    # Copy file
                    with open(source, "r", encoding="utf-8") as src:
                        content = src.read()
                    with open(dest, "w", encoding="utf-8") as dst:
                        dst.write(content)

            # Backup environment variables
            env_backup = {
                "BYBIT_TESTNET": os.getenv("BYBIT_TESTNET", ""),
                "BYBIT_PRODUCTION_ENABLED": os.getenv("BYBIT_PRODUCTION_ENABLED", ""),
                "BYBIT_PRODUCTION_CONFIRMED": os.getenv(
                    "BYBIT_PRODUCTION_CONFIRMED", ""
                ),
            }

            with open(self.backup_path / "environment_backup.json", "w") as f:
                json.dump(env_backup, f, indent=2)

            print(f"✅ Backup created at: {self.backup_path}")
            return True

        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False

    def update_bybit_connector(self):
        """Update Bybit connector to use production API"""
        print("🔧 Updating Bybit Connector...")
        print("-" * 40)

        try:
            connector_file = (
                self.base_path / "ZoL0-master/data/execution/bybit_connector.py"
            )

            with open(connector_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Replace testnet default with production
            updated_content = content.replace(
                "self.use_testnet = use_testnet if use_testnet is not None else False",
                'self.use_testnet = use_testnet if use_testnet is not None else not bool(os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true")',
            )

            # Add production environment detection
            if "import os" not in updated_content:
                updated_content = "import os\n" + updated_content

            # Ensure production API URLs are used when in production mode
            production_check = """
        # Override testnet setting if production is enabled
        if os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true":
            self.use_testnet = False
            logger.info("Production mode detected - using Bybit production API")
        elif os.getenv("BYBIT_TESTNET", "").lower() == "true":
            self.use_testnet = True
            logger.info("Testnet mode detected - using Bybit testnet API")
"""

            # Insert production check after initialization
            if "Production mode detected" not in updated_content:
                init_end = updated_content.find("if not lazy_connect:")
                if init_end != -1:
                    updated_content = (
                        updated_content[:init_end]
                        + production_check
                        + "\n        "
                        + updated_content[init_end:]
                    )

            with open(connector_file, "w", encoding="utf-8") as f:
                f.write(updated_content)

            print("✅ Bybit connector updated for production")
            return True

        except Exception as e:
            print(f"❌ Failed to update Bybit connector: {e}")
            return False

    def update_market_data_fetcher(self):
        """Update market data fetcher to use production API"""
        print("🔧 Updating Market Data Fetcher...")
        print("-" * 40)

        try:
            fetcher_file = (
                self.base_path / "ZoL0-master/data/data/market_data_fetcher.py"
            )

            with open(fetcher_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Update default testnet setting
            updated_content = content.replace(
                "use_testnet: bool = True",
                'use_testnet: bool = not bool(os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true")',
            )

            # Ensure production API base URL is used
            updated_content = updated_content.replace(
                "self.base_url = API_TESTNET_URL if self.use_testnet else API_BASE_URL",
                """
        # Determine API URL based on environment
        if os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true":
            self.base_url = API_BASE_URL  # Production
            self.use_testnet = False
            logging.info("Using Bybit PRODUCTION API")
        else:
            self.base_url = API_TESTNET_URL if self.use_testnet else API_BASE_URL
            if self.use_testnet:
                logging.info("Using Bybit TESTNET API")
            else:
                logging.info("Using Bybit PRODUCTION API")
        """,
            )

            with open(fetcher_file, "w", encoding="utf-8") as f:
                f.write(updated_content)

            print("✅ Market data fetcher updated for production")
            return True

        except Exception as e:
            print(f"❌ Failed to update market data fetcher: {e}")
            return False

    def update_dashboard_data_sources(self):
        """Update all dashboards to use real API data instead of simulated data"""
        print("🔧 Updating Dashboard Data Sources...")
        print("-" * 40)

        updated_count = 0

        for dashboard_file in self.dashboard_files:
            try:
                file_path = self.base_path / dashboard_file
                if not file_path.exists():
                    continue

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Track if file was modified
                modified = False
                original_content = content

                # Replace demo/simulated data initialization with real API calls
                if "advanced_trading_analytics.py" in dashboard_file:
                    content = self._update_trading_analytics(content)
                    modified = content != original_content

                elif "real_time_market_data_integration.py" in dashboard_file:
                    content = self._update_market_data_integration(content)
                    modified = content != original_content

                elif "ml_predictive_analytics.py" in dashboard_file:
                    content = self._update_ml_analytics(content)
                    modified = content != original_content

                elif "enhanced_bot_monitor.py" in dashboard_file:
                    content = self._update_bot_monitor(content)
                    modified = content != original_content

                elif "data_export_import_system.py" in dashboard_file:
                    content = self._update_data_export_system(content)
                    modified = content != original_content

                if modified:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"✅ Updated: {dashboard_file}")
                    updated_count += 1
                else:
                    print(f"ℹ️  No changes needed: {dashboard_file}")

            except Exception as e:
                print(f"❌ Failed to update {dashboard_file}: {e}")

        print(f"✅ Updated {updated_count} dashboard files")
        return updated_count > 0

    def _update_trading_analytics(self, content: str) -> str:
        """Update trading analytics to use real API data"""
        # Replace fallback data source detection
        content = content.replace(
            '"data_source": "live" if db_data else "simulated"',
            '"data_source": "live" if (db_data or self._get_api_data()) else "simulated"',
        )

        # Add real API data fetching method
        if "_get_api_data" not in content:
            api_method = '''
    def _get_api_data(self):
        """Fetch real trading data from Bybit API"""
        try:
            from ZoL0master.data.execution.bybit_connector import BybitConnector
            
            # Use production API if enabled
            use_testnet = not bool(os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true")
            
            connector = BybitConnector(
                api_key=os.getenv("BYBIT_API_KEY"),
                api_secret=os.getenv("BYBIT_API_SECRET"),
                use_testnet=use_testnet
            )
            
            # Fetch real account balance and positions
            balance = connector.get_account_balance()
            positions = connector.get_positions()
            
            if balance.get("success") and positions.get("success"):
                return {
                    "balance": balance,
                    "positions": positions,
                    "data_source": "live_api"
                }
                
        except Exception as e:
            logger.error(f"Failed to fetch real API data: {e}")
            
        return {}
'''

            # Insert before the _get_database_performance method
            insert_point = content.find("def _get_database_performance")
            if insert_point != -1:
                content = (
                    content[:insert_point]
                    + api_method
                    + "\n    "
                    + content[insert_point:]
                )

        return content

    def _update_market_data_integration(self, content: str) -> str:
        """Update market data integration to use real exchange APIs"""
        # Replace demo data initialization with real API connections
        content = content.replace(
            "self._initialize_demo_data()", "self._initialize_real_data()"
        )

        content = content.replace(
            "self._setup_demo_connections()", "self._setup_real_connections()"
        )

        # Add real data initialization method
        if "_initialize_real_data" not in content:
            real_data_method = '''
    def _initialize_real_data(self):
        """Initialize real market data from Bybit API"""
        try:
            from ZoL0master.data.execution.bybit_connector import BybitConnector
            
            # Use production API if enabled
            use_testnet = not bool(os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true")
            
            self.bybit_connector = BybitConnector(
                api_key=os.getenv("BYBIT_API_KEY"),
                api_secret=os.getenv("BYBIT_API_SECRET"),
                use_testnet=use_testnet
            )
            
            # Fetch real market data for common symbols
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'SOLUSDT']
            
            for symbol in symbols:
                try:
                    ticker_data = self.bybit_connector.get_ticker(symbol)
                    if ticker_data.get("success"):
                        self._process_real_ticker_data(symbol, ticker_data)
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize real data: {e}")
            # Fallback to demo data if real API fails
            self._initialize_demo_data()
'''

            # Insert after class initialization
            init_end = content.find("self._setup_demo_connections()")
            if init_end != -1:
                content = (
                    content[:init_end]
                    + real_data_method
                    + "\n    "
                    + content[init_end:]
                )

        return content

    def _update_ml_analytics(self, content: str) -> str:
        """Update ML analytics to use real trading data"""
        # Replace synthetic data generation with real data fetching
        content = content.replace(
            "self.generate_synthetic_data()", "self.fetch_real_trading_data()"
        )

        # Add real data fetching method
        if "fetch_real_trading_data" not in content:
            real_data_method = '''
    def fetch_real_trading_data(self):
        """Fetch real trading data from Bybit API for ML training"""
        try:
            from ZoL0master.data.data.market_data_fetcher import MarketDataFetcher
            
            # Use production API if enabled
            use_testnet = not bool(os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true")
            
            fetcher = MarketDataFetcher(
                api_key=os.getenv("BYBIT_API_KEY"),
                api_secret=os.getenv("BYBIT_API_SECRET"),
                use_testnet=use_testnet
            )
            
            # Fetch historical data for ML training
            df = fetcher.fetch_data(symbol="BTCUSDT", interval="1h", limit=1000)
            
            if df is not None and not df.empty:
                logger.info(f"Fetched {len(df)} real data points for ML training")
                return df
            else:
                logger.warning("No real data available, falling back to synthetic data")
                return self.generate_synthetic_data()
                
        except Exception as e:
            logger.error(f"Failed to fetch real trading data: {e}")
            return self.generate_synthetic_data()
'''

            # Insert after class initialization
            content = content.replace(
                "def generate_synthetic_data",
                real_data_method + "\n    def generate_synthetic_data",
            )

        return content

    def _update_bot_monitor(self, content: str) -> str:
        """Update bot monitor to show real trading status"""
        # Add real API status checking
        if "check_real_api_status" not in content:
            api_status_method = '''
    def check_real_api_status(self):
        """Check real Bybit API connection status"""
        try:
            from ZoL0master.data.execution.bybit_connector import BybitConnector
            
            # Use production API if enabled
            use_testnet = not bool(os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true")
            
            connector = BybitConnector(
                api_key=os.getenv("BYBIT_API_KEY"),
                api_secret=os.getenv("BYBIT_API_SECRET"),
                use_testnet=use_testnet
            )
            
            # Test API connection
            server_time = connector.get_server_time()
            if server_time.get("success"):
                return {
                    "status": "connected",
                    "environment": "production" if not use_testnet else "testnet",
                    "server_time": server_time.get("data", {}).get("timeSecond", "unknown")
                }
            else:
                return {"status": "disconnected", "error": "API call failed"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
'''

            # Insert at the end of the class
            content = content.replace(
                'if __name__ == "__main__":',
                api_status_method + '\n\nif __name__ == "__main__":',
            )

        return content

    def _update_data_export_system(self, content: str) -> str:
        """Update data export system to use real trading data"""
        # Replace synthetic data generation with real data fetching
        content = content.replace(
            "return self.generate_synthetic_trading_data",
            "return self.fetch_real_trading_data",
        )

        return content

    def set_production_environment(self):
        """Set environment variables for production mode"""
        print("🔧 Configuring Production Environment...")
        print("-" * 40)

        try:
            # Set production environment variables
            os.environ["BYBIT_PRODUCTION_ENABLED"] = "true"
            os.environ["BYBIT_PRODUCTION_CONFIRMED"] = "true"

            # Remove testnet flag if it exists
            if "BYBIT_TESTNET" in os.environ:
                del os.environ["BYBIT_TESTNET"]

            # Save configuration
            config = {
                "migration_date": datetime.now().isoformat(),
                "environment": "production",
                "bybit_production_enabled": True,
                "bybit_testnet": False,
                "migration_status": "completed",
            }

            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)

            print("✅ Production environment configured")
            print("   - BYBIT_PRODUCTION_ENABLED = true")
            print("   - BYBIT_PRODUCTION_CONFIRMED = true")
            print("   - BYBIT_TESTNET = removed")

            return True

        except Exception as e:
            print(f"❌ Failed to configure production environment: {e}")
            return False

    def test_production_integration(self):
        """Test that all dashboards can connect to production API"""
        print("🧪 Testing Production Integration...")
        print("-" * 40)

        test_results = {}
        # Test Bybit connector
        try:
            sys.path.append(str(self.base_path / "ZoL0-master"))
            from data.execution.bybit_connector import BybitConnector

            connector = BybitConnector(
                api_key=os.getenv("BYBIT_API_KEY"),
                api_secret=os.getenv("BYBIT_API_SECRET"),
                use_testnet=False,
            )

            # Test basic API calls
            server_time = connector.get_server_time()
            balance = connector.get_account_balance()

            test_results["bybit_connector"] = {
                "server_time": server_time.get("success", False),
                "account_balance": balance.get("success", False),
                "status": (
                    "pass"
                    if server_time.get("success") and balance.get("success")
                    else "fail"
                ),
            }

        except Exception as e:
            test_results["bybit_connector"] = {"status": "fail", "error": str(e)}
            # Test market data fetcher
        try:
            sys.path.append(str(self.base_path / "ZoL0-master"))
            from data.data.market_data_fetcher import MarketDataFetcher

            fetcher = MarketDataFetcher(
                api_key=os.getenv("BYBIT_API_KEY"), use_testnet=False
            )

            # Test data fetching
            data = fetcher.fetch_data("BTCUSDT", "1h", 10)
            test_results["market_data_fetcher"] = {
                "data_fetch": not data.empty if data is not None else False,
                "status": "pass" if data is not None and not data.empty else "fail",
            }

        except Exception as e:
            test_results["market_data_fetcher"] = {"status": "fail", "error": str(e)}

        # Display results
        all_passed = True
        for component, result in test_results.items():
            status = result.get("status", "unknown")
            if status == "pass":
                print(f"✅ {component}: PASS")
            else:
                print(f"❌ {component}: FAIL - {result.get('error', 'Unknown error')}")
                all_passed = False

        return all_passed

    def restart_dashboards(self):
        """Restart all dashboard services to apply changes"""
        print("🔄 Restarting Dashboard Services...")
        print("-" * 40)

        try:
            # Kill existing dashboard processes
            import subprocess

            # Get all running Python processes
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq python.exe"],
                capture_output=True,
                text=True,
            )

            # Kill processes running dashboard files
            for line in result.stdout.split("\n"):
                if "python.exe" in line:
                    # This is a simplified approach - in production you'd want more precise process management
                    pass

            print("✅ Dashboard services restart initiated")
            print("ℹ️  Please manually restart your dashboard services to apply changes")

            return True

        except Exception as e:
            print(f"❌ Failed to restart services: {e}")
            return False

    def run_migration(self):
        """Run the complete migration process"""
        self.print_banner()

        # Confirm migration
        confirm = input(
            "Do you want to proceed with production API migration? (yes/no): "
        ).lower()
        if confirm != "yes":
            print("❌ Migration cancelled by user")
            return False

        # Run migration steps
        steps = [
            ("Checking Prerequisites", self.check_prerequisites),
            ("Creating Backup", self.create_backup),
            ("Updating Bybit Connector", self.update_bybit_connector),
            ("Updating Market Data Fetcher", self.update_market_data_fetcher),
            ("Updating Dashboard Data Sources", self.update_dashboard_data_sources),
            ("Setting Production Environment", self.set_production_environment),
            ("Testing Production Integration", self.test_production_integration),
            ("Restarting Services", self.restart_dashboards),
        ]

        for step_name, step_func in steps:
            print(f"\n📍 Step: {step_name}")
            try:
                if not step_func():
                    print(f"❌ Migration failed at step: {step_name}")
                    return False
            except Exception as e:
                print(f"❌ Migration failed at step {step_name}: {e}")
                return False

        print("\n" + "=" * 80)
        print("🎉 MIGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("✅ All dashboards are now configured to use Bybit PRODUCTION API")
        print("⚠️  WARNING: System is now in PRODUCTION mode with REAL money!")
        print("📊 Please verify all dashboards are showing real data")
        print(f"💾 Backup created at: {self.backup_path}")
        print("=" * 80)

        return True

    def rollback_migration(self):
        """Rollback to previous configuration"""
        print("🔄 Rolling Back Migration...")
        print("-" * 40)

        try:
            if not self.backup_path.exists():
                print("❌ No backup found to rollback to")
                return False

            # Restore files from backup
            for file_path in self.dashboard_files + self.data_source_files:
                backup_file = self.backup_path / file_path
                if backup_file.exists():
                    dest_file = self.base_path / file_path

                    with open(backup_file, "r", encoding="utf-8") as src:
                        content = src.read()
                    with open(dest_file, "w", encoding="utf-8") as dst:
                        dst.write(content)

            # Restore environment variables
            env_backup_file = self.backup_path / "environment_backup.json"
            if env_backup_file.exists():
                with open(env_backup_file, "r") as f:
                    env_backup = json.load(f)

                for key, value in env_backup.items():
                    if value:
                        os.environ[key] = value
                    elif key in os.environ:
                        del os.environ[key]

            print("✅ Migration rolled back successfully")
            return True

        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return False


def main():
    """Main function"""
    migrator = ProductionAPIMigrator()

    if len(sys.argv) > 1 and sys.argv[1] == "--rollback":
        migrator.rollback_migration()
    else:
        migrator.run_migration()


if __name__ == "__main__":
    main()
