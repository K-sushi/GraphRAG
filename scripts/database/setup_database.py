#!/usr/bin/env python3
"""
Database setup script for GraphRAG Implementation.
Initializes PostgreSQL database with proper schemas, tables, and sample data.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import asyncpg
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Database setup and initialization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.admin_config = {
            'host': config['host'],
            'port': config['port'],
            'user': config.get('admin_user', 'postgres'),
            'password': config.get('admin_password', config['password']),
            'database': 'postgres'  # Connect to default database first
        }
    
    def _get_connection_string(self, database: str = None, user: str = None) -> str:
        """Generate connection string."""
        db = database or self.config['database']
        username = user or self.config['user']
        password = self.config['password']
        host = self.config['host']
        port = self.config['port']
        
        return f"postgresql://{username}:{password}@{host}:{port}/{db}"
    
    async def check_connection(self) -> bool:
        """Test database connection."""
        try:
            conn = await asyncpg.connect(**self.admin_config)
            await conn.close()
            logger.info("✅ Database connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def create_database_and_users(self) -> bool:
        """Create database and users using psycopg2 (synchronous)."""
        try:
            # Connect as admin user
            conn = psycopg2.connect(**self.admin_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Create database
            database_name = self.config['database']
            logger.info(f"Creating database: {database_name}")
            
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
            if not cursor.fetchone():
                cursor.execute(f'CREATE DATABASE "{database_name}"')
                logger.info(f"✅ Database '{database_name}' created")
            else:
                logger.info(f"ℹ️  Database '{database_name}' already exists")
            
            # Create users
            users_to_create = ['lightrag', 'n8n']
            for username in users_to_create:
                user_password = self.config.get(f'{username}_password', self.config['password'])
                
                cursor.execute("SELECT 1 FROM pg_user WHERE usename = %s", (username,))
                if not cursor.fetchone():
                    cursor.execute(f'CREATE USER "{username}" WITH PASSWORD %s', (user_password,))
                    logger.info(f"✅ User '{username}' created")
                else:
                    logger.info(f"ℹ️  User '{username}' already exists")
                    # Update password
                    cursor.execute(f'ALTER USER "{username}" WITH PASSWORD %s', (user_password,))
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create database and users: {e}")
            return False
    
    async def run_sql_file(self, sql_file_path: Path, database: str = None) -> bool:
        """Execute SQL file."""
        try:
            if not sql_file_path.exists():
                logger.error(f"❌ SQL file not found: {sql_file_path}")
                return False
            
            # Read SQL file
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Connect to target database
            db_name = database or self.config['database']
            conn_config = {
                'host': self.config['host'],
                'port': self.config['port'],
                'user': self.config.get('admin_user', 'postgres'),
                'password': self.config.get('admin_password', self.config['password']),
                'database': db_name
            }
            
            conn = await asyncpg.connect(**conn_config)
            
            logger.info(f"📄 Executing SQL file: {sql_file_path.name}")
            
            # Split and execute SQL statements
            # Handle multi-statement SQL properly
            await conn.execute(sql_content)
            
            await conn.close()
            logger.info(f"✅ SQL file executed successfully: {sql_file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to execute SQL file {sql_file_path}: {e}")
            return False
    
    async def create_extensions(self) -> bool:
        """Create required PostgreSQL extensions."""
        try:
            conn_config = {
                'host': self.config['host'],
                'port': self.config['port'],
                'user': self.config.get('admin_user', 'postgres'),
                'password': self.config.get('admin_password', self.config['password']),
                'database': self.config['database']
            }
            
            conn = await asyncpg.connect(**conn_config)
            
            extensions = [
                'uuid-ossp',
                'vector',
                'pg_trgm',
                'btree_gin'
            ]
            
            for ext in extensions:
                try:
                    await conn.execute(f'CREATE EXTENSION IF NOT EXISTS "{ext}"')
                    logger.info(f"✅ Extension '{ext}' created/verified")
                except Exception as e:
                    logger.warning(f"⚠️  Could not create extension '{ext}': {e}")
            
            await conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create extensions: {e}")
            return False
    
    async def verify_setup(self) -> bool:
        """Verify database setup."""
        try:
            conn_config = {
                'host': self.config['host'],
                'port': self.config['port'],
                'user': 'lightrag',
                'password': self.config.get('lightrag_password', self.config['password']),
                'database': self.config['database']
            }
            
            conn = await asyncpg.connect(**conn_config)
            
            # Check schemas
            schemas = await conn.fetch("SELECT schema_name FROM information_schema.schemata WHERE schema_name IN ('lightrag', 'n8n', 'shared')")
            schema_names = [row['schema_name'] for row in schemas]
            
            expected_schemas = ['lightrag', 'n8n', 'shared']
            for schema in expected_schemas:
                if schema in schema_names:
                    logger.info(f"✅ Schema '{schema}' exists")
                else:
                    logger.error(f"❌ Schema '{schema}' missing")
                    return False
            
            # Check key tables
            tables_check = await conn.fetch("""
                SELECT table_schema, table_name 
                FROM information_schema.tables 
                WHERE table_schema IN ('lightrag', 'n8n', 'shared')
                ORDER BY table_schema, table_name
            """)
            
            logger.info(f"📊 Found {len(tables_check)} tables across all schemas")
            
            # Check extensions
            extensions = await conn.fetch("SELECT extname FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm', 'btree_gin')")
            ext_names = [row['extname'] for row in extensions]
            
            for ext in ['vector', 'uuid-ossp', 'pg_trgm', 'btree_gin']:
                if ext in ext_names:
                    logger.info(f"✅ Extension '{ext}' installed")
                else:
                    logger.warning(f"⚠️  Extension '{ext}' not found")
            
            # Test vector functionality
            try:
                await conn.execute("SELECT '[1,2,3]'::vector(3)")
                logger.info("✅ Vector extension working")
            except Exception as e:
                logger.error(f"❌ Vector extension test failed: {e}")
            
            await conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup verification failed: {e}")
            return False
    
    async def insert_sample_data(self) -> bool:
        """Insert sample data for testing."""
        try:
            conn_config = {
                'host': self.config['host'],
                'port': self.config['port'],
                'user': 'lightrag',
                'password': self.config.get('lightrag_password', self.config['password']),
                'database': self.config['database']
            }
            
            conn = await asyncpg.connect(**conn_config)
            
            # Check if sample data already exists
            count = await conn.fetchval("SELECT COUNT(*) FROM lightrag.entities")
            if count > 0:
                logger.info("ℹ️  Sample data already exists")
                await conn.close()
                return True
            
            # Insert sample entities
            entities = [
                ('Artificial Intelligence', 'concept', 'The simulation of human intelligence in machines', 0.95),
                ('Machine Learning', 'concept', 'A subset of AI that enables computers to learn without explicit programming', 0.90),
                ('Deep Learning', 'concept', 'A subset of machine learning based on artificial neural networks', 0.88),
                ('Neural Networks', 'concept', 'Computing systems inspired by biological neural networks', 0.85),
                ('Natural Language Processing', 'concept', 'AI technology for understanding human language', 0.87),
                ('Computer Vision', 'concept', 'AI technology for interpreting visual information', 0.82)
            ]
            
            entity_ids = []
            for name, type_, desc, confidence in entities:
                entity_id = await conn.fetchval(
                    "INSERT INTO lightrag.entities (name, type, description, confidence_score) VALUES ($1, $2, $3, $4) RETURNING entity_id",
                    name, type_, desc, confidence
                )
                entity_ids.append((name, entity_id))
            
            logger.info(f"✅ Inserted {len(entity_ids)} sample entities")
            
            # Create entity ID mapping
            entity_map = {name: id_ for name, id_ in entity_ids}
            
            # Insert sample relationships
            relationships = [
                ('Machine Learning', 'Artificial Intelligence', 'part_of', 0.8, 0.85),
                ('Deep Learning', 'Machine Learning', 'part_of', 0.9, 0.88),
                ('Deep Learning', 'Neural Networks', 'uses', 0.7, 0.80),
                ('Natural Language Processing', 'Artificial Intelligence', 'part_of', 0.8, 0.87),
                ('Computer Vision', 'Artificial Intelligence', 'part_of', 0.7, 0.82),
                ('Deep Learning', 'Natural Language Processing', 'enables', 0.6, 0.75),
                ('Deep Learning', 'Computer Vision', 'enables', 0.6, 0.75)
            ]
            
            for source, target, rel_type, weight, confidence in relationships:
                if source in entity_map and target in entity_map:
                    await conn.execute("""
                        INSERT INTO lightrag.relationships 
                        (source_entity_id, target_entity_id, relationship_type, weight, confidence_score)
                        VALUES ($1, $2, $3, $4, $5)
                    """, entity_map[source], entity_map[target], rel_type, weight, confidence)
            
            logger.info(f"✅ Inserted {len(relationships)} sample relationships")
            
            # Insert sample documents
            documents = [
                "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.",
                "Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
                "Deep Learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
                "Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."
            ]
            
            for i, doc in enumerate(documents):
                await conn.execute("""
                    INSERT INTO n8n.documents_v2 
                    (content, content_hash, file_id, metadata)
                    VALUES ($1, $2, $3, $4)
                """, doc, f"hash_{i}", f"doc_{i}", '{"type": "sample", "source": "setup"}')
            
            logger.info(f"✅ Inserted {len(documents)} sample documents")
            
            await conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert sample data: {e}")
            return False
    
    async def run_full_setup(self, include_sample_data: bool = True) -> bool:
        """Run complete database setup."""
        logger.info("🚀 Starting GraphRAG database setup...")
        
        # Step 1: Check connection
        if not await self.check_connection():
            return False
        
        # Step 2: Create database and users
        if not self.create_database_and_users():
            return False
        
        # Step 3: Create extensions
        if not await self.create_extensions():
            return False
        
        # Step 4: Run initialization SQL
        script_dir = Path(__file__).parent
        sql_file = script_dir / "init_postgres.sql"
        
        if not await self.run_sql_file(sql_file):
            return False
        
        # Step 5: Insert sample data
        if include_sample_data:
            if not await self.insert_sample_data():
                logger.warning("⚠️  Sample data insertion failed, but setup can continue")
        
        # Step 6: Verify setup
        if not await self.verify_setup():
            return False
        
        logger.info("🎉 GraphRAG database setup completed successfully!")
        return True

def load_config_from_env() -> Dict[str, Any]:
    """Load database configuration from environment variables."""
    return {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'database': os.getenv('POSTGRES_DB', 'lightrag_production'),
        'user': os.getenv('POSTGRES_USER', 'lightrag'),
        'password': os.getenv('POSTGRES_PASSWORD', 'your_password_here'),
        'admin_user': os.getenv('POSTGRES_ADMIN_USER', 'postgres'),
        'admin_password': os.getenv('POSTGRES_ADMIN_PASSWORD', os.getenv('POSTGRES_PASSWORD', 'your_password_here')),
        'lightrag_password': os.getenv('LIGHTRAG_PASSWORD', os.getenv('POSTGRES_PASSWORD', 'your_password_here')),
        'n8n_password': os.getenv('N8N_PASSWORD', os.getenv('POSTGRES_PASSWORD', 'your_password_here'))
    }

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Setup GraphRAG PostgreSQL database')
    parser.add_argument('--no-sample-data', action='store_true', help='Skip inserting sample data')
    parser.add_argument('--config-file', help='Path to configuration file (JSON)')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing setup')
    parser.add_argument('--host', help='Database host')
    parser.add_argument('--port', type=int, help='Database port')
    parser.add_argument('--database', help='Database name')
    parser.add_argument('--user', help='Database user')
    parser.add_argument('--password', help='Database password')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_env()
    
    # Override with command line arguments
    if args.host:
        config['host'] = args.host
    if args.port:
        config['port'] = args.port
    if args.database:
        config['database'] = args.database
    if args.user:
        config['user'] = args.user
    if args.password:
        config['password'] = args.password
    
    # Initialize setup
    setup = DatabaseSetup(config)
    
    logger.info(f"Database configuration:")
    logger.info(f"  Host: {config['host']}")
    logger.info(f"  Port: {config['port']}")
    logger.info(f"  Database: {config['database']}")
    logger.info(f"  User: {config['user']}")
    
    try:
        if args.verify_only:
            # Only verify setup
            success = await setup.verify_setup()
        else:
            # Run full setup
            success = await setup.run_full_setup(
                include_sample_data=not args.no_sample_data
            )
        
        if success:
            logger.info("✅ Database setup completed successfully!")
            
            # Print connection information
            print(f"\n{'='*60}")
            print("📊 DATABASE CONNECTION INFORMATION")
            print(f"{'='*60}")
            print(f"Host: {config['host']}")
            print(f"Port: {config['port']}")
            print(f"Database: {config['database']}")
            print(f"LightRAG User: lightrag")
            print(f"n8n User: n8n")
            
            print(f"\n📝 CONNECTION STRINGS:")
            print(f"LightRAG: postgresql://lightrag:***@{config['host']}:{config['port']}/{config['database']}")
            print(f"n8n: postgresql://n8n:***@{config['host']}:{config['port']}/{config['database']}")
            
            print(f"\n🎯 NEXT STEPS:")
            print("1. Update your application configuration with these connection details")
            print("2. Test the connection with your LightRAG server")
            print("3. Configure n8n workflows to use the database")
            print("4. Monitor the database with the provided views and functions")
            
            sys.exit(0)
        else:
            logger.error("❌ Database setup failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⚠️  Setup interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Setup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())