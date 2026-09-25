


# Architectural Decision Record (ADR)

## Database ORM

### Context and Problem Statement

We need an ORM to interface with our PostGreSQL database to manage data on patients, chat and voice sessions, and clinical information. 

### Decision Drivers

- Scalability
- Data consistency
- Ease of integration with existing services

### Considered Options

- Prisma 
- SQLALchemy
- SQLite

### Decision Outcome

We chose SQLAlchemy because:
- Industry standard in Python
- First class FastAPI support
- Great async support
- Extremely flexible for complex queries
- Easy to integrate with ML/data pipelines

We are using FastAPI/Python on the backend, AI/ML, complex data models, and healthcare integration, which makes SQLAlchemy the best choice.

