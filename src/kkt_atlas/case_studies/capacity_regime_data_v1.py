"""Four physical resource-demand regimes driven by two contextual signals."""
import hashlib
from dataclasses import replace
import numpy as np
from kkt_atlas.data import ObservedSplit
from kkt_atlas.confirmation_data_v1 import ConditionalTestData

NAMESPACE='capacity-four-resource-regimes-v1'


def regime_data(fitting,evaluation,service,total_capacity,*,noise=.035):
    service=np.asarray(service,dtype=np.float64)
    if fitting.context_dim<2 or fitting.support_size%4 or service.shape[0]!=fitting.scenario_dim or service.shape[1]<3:
        raise ValueError('Need two context signals, four balanced regimes and at least three resources')
    if not np.isfinite(noise) or noise<0 or not np.isfinite(total_capacity) or total_capacity<=0:
        raise ValueError('Invalid demand scale or noise')
    if not np.isfinite(service).all() or np.any(service<0) or not np.allclose(service.sum(1),1):
        raise ValueError('Service rows must be convex resource mixtures')
    if fitting.catalog_hash!=evaluation.catalog_hash or fitting.metadata!=evaluation.metadata:
        raise ValueError('Base fitting and evaluator data do not match')
    seed=int(fitting.metadata['seed'])
    def rng(tag):return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{NAMESPACE}|{seed}|{tag}'.encode()).digest()[:8],'little'))
    signs=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    directions,_=np.linalg.qr(rng('context-directions').normal(size=(fitting.context_dim,2)))
    contrasts=rng('resource-directions').normal(size=(service.shape[1],2));contrasts-=contrasts.mean(0)
    resource_basis,_=np.linalg.qr(contrasts)
    def softmax(values):
        exp=np.exp(values-values.max(1,keepdims=True));return exp/exp.sum(1,keepdims=True)
    required=total_capacity*softmax(signs@resource_basis.T)
    centers=required@service.T
    labels=np.repeat(np.arange(4),fitting.support_size//4)
    perturbation=rng('within-regime-demand').normal(size=fitting.catalog.shape)
    catalog=centers[labels]*np.exp(noise*perturbation-.5*noise**2)
    def probabilities(contexts):
        weights=softmax(contexts@directions@signs.T)
        return weights[:,labels]/(fitting.support_size//4)
    def observed(split,name):
        probabilities0=probabilities(split.contexts)
        cdf=np.cumsum(probabilities0,axis=1);cdf[:,-1]=1.
        indices=np.sum(rng(name).random(len(cdf))[:,None]>cdf,axis=1)
        return ObservedSplit(split.contexts,indices)
    metadata=dict(schema=NAMESPACE,seed=seed,context_dim=fitting.context_dim,scenario_dim=fitting.scenario_dim,
        support_size=fitting.support_size,training_size=len(fitting.training.contexts),validation_size=len(fitting.validation.contexts),
        test_size=len(evaluation.contexts),regime_count=4,signal_dimension=2,idiosyncratic_scale=noise,
        resource_contrast_scale=1.,context_logit_scale=1.,conditional_map='affine_softmax_regime_probabilities',
        design='Four combinations of two exogenous demand states, fixed physical service technology, positive within-state demand variation.')
    new=replace(fitting,catalog=catalog,training=observed(fitting.training,'training-outcomes'),
        validation=observed(fitting.validation,'validation-outcomes'),metadata=metadata)
    test=ConditionalTestData(evaluation.contexts,probabilities(evaluation.contexts),new.catalog_hash,metadata)
    return new,test,dict(context_directions=directions,resource_basis=resource_basis,regime_centers=centers,
        regime_labels=labels,resource_requirements=required,logit_signs=signs)
