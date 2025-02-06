class DatabaseRouter:
    """
    A router to control all database operations on models for different databases.
    """

    def db_for_read(self, model, **hints):
        """Point read operations to the correct database."""
        if model._meta.app_label == 'DSAI':
            return 'dsai'
        return 'default'

    def db_for_write(self, model, **hints):
        """Point write operations to the correct database."""
        if model._meta.app_label == 'DSAI':
            return 'dsai'
        return 'default'

    def allow_relation(self, obj1, obj2, **hints):
        """Allow relationships between objects in the same database."""
        if obj1._state.db == obj2._state.db:
            return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """Ensure that apps only appear in the correct database."""
        if app_label == 'DSAI':
            return db == 'dsai'
        return db == 'default'
