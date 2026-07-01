from nomad.config.models.plugins import SchemaPackageEntryPoint


class BayesianOptimizationPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_bayesian_optimization.schema_packages.bayesian_optimization import (
            m_package,
        )

        return m_package


class CVDPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_bayesian_optimization.schema_packages.cvd import m_package

        return m_package


cvd = CVDPackageEntryPoint(
    name='CVD',
    description='Schema package for chemical vapor deposition experiments.',
)

bayesian_optimization = BayesianOptimizationPackageEntryPoint(
    name='Bayesian Optimization',
    description='Schema package for Bayesian optimization runs.',
)
