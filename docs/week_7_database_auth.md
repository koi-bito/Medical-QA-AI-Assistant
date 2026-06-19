# Week 7: Database & Authentication

This week kicked off **Phase 2: Full-Stack AI Product**, transforming the ML demo into a robust, production-quality web application backend.

## Key Learnings

### MySQL Database + SQLAlchemy Setup (Day 46)
- **SQLAlchemy ORM:** Using an Object-Relational Mapper (ORM) like SQLAlchemy allows us to interact with the database using Python objects rather than writing raw SQL strings. This prevents SQL injection and makes database schemas easier to manage and migrate.
- **Connection Strings & Passwords:** The `DATABASE_URL` format used by SQLAlchemy is sensitive to special characters. For example, if a database password contains an `@` symbol, it must be URL-encoded (as `%40`) so SQLAlchemy doesn't misinterpret it as the `@` separating the credentials from the host address (resulting in `getaddrinfo failed`).
- **Dependencies & Environment:** Using `pymysql` provides a pure-Python MySQL driver that requires no C-extension compilation, keeping the setup lightweight. Storing the database credentials securely in a `.env` file via `python-dotenv` ensures no secrets are hardcoded in the source files.
