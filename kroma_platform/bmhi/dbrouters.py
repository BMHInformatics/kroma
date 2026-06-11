class DSAppRouter:

    app_label = 'DSapp'
    db_alias = 'dsai'

    def db_for_read(self, model, **hints):
        if model._meta.app_label == self.app_label:
            return self.db_alias
        return None

    def db_for_write(self, model, **hints):
        if model._meta.app_label == self.app_label:
            return self.db_alias
        return None

    def allow_relation(self, obj1, obj2, **hints):
        if (obj1._meta.app_label == self.app_label and
                obj2._meta.app_label == self.app_label):
            return True
        if (obj1._meta.app_label != self.app_label and
                obj2._meta.app_label != self.app_label):
            return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if app_label == self.app_label:
            return db == self.db_alias

        if db == self.db_alias:
            return False

        return None
