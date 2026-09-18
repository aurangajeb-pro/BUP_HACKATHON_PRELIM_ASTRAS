"""Regression tests for failures reproduced during review; no live model required."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

import app.llm as llm
from app.optimizer import SolverFailure, optimize
from app.schemas import Battery, OptimizeRequest


@pytest.fixture
def payload():
    return json.loads((Path(__file__).resolve().parents[1] / 'sample_input.json').read_text())


def noop(index=0):
    return {'note_index': index, 'applies': False, 'directive_type': 'no_op',
            'structured_adjustment': None, 'explanation': 'No energy instruction.'}


def active(kind='no_charge_window', **fields):
    return {'note_index': 0, 'applies': True, 'directive_type': kind,
            'structured_adjustment': {'kind': kind, 'hours': [2, 3], **fields},
            'explanation': 'Apply the requested restriction.'}


def model_output(entries):
    return json.dumps({'interpretations': entries})


@pytest.mark.parametrize('path,value', [
    (('battery', 'initial_energy_kwh'), 0),
    (('battery', 'minimum_energy_kwh'), 10000),
    (('battery', 'capacity_kwh'), float('inf')),
    (('battery', 'max_charge_kwh_per_hour'), float('nan')),
    (('hours', 0, 'demand_kwh'), float('inf')),
    (('hours', 0, 'demand_kwh'), -1),
    (('hours', 0, 'hour'), True),
    (('hours', 0, 'hour'), '0'),
    (('operator_notes',), []),
    (('operator_notes',), [' ']),
    (('operator_notes',), ['a', 'b', 'c', 'd']),
    (('scenario_id',), 123),
])
def test_invalid_inputs_return_controlled_400(client, payload, path, value):
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    response = client.post('/optimize-energy', content=json.dumps(payload),
                           headers={'content-type': 'application/json'})
    assert response.status_code == 400
    assert response.json()['error'] == 'invalid_request'


def test_invalid_json(client):
    assert client.post('/optimize-energy', content='{',
                       headers={'content-type': 'application/json'}).status_code == 400


def test_duplicate_hours(client, payload):
    payload['hours'][1]['hour'] = 0
    assert client.post('/optimize-energy', json=payload).status_code == 400


def test_shuffled_hours_preserve_physical_schedule(client, payload, no_directives):
    first = client.post('/optimize-energy', json=payload)
    payload['hours'].reverse()
    shuffled = client.post('/optimize-energy', json=payload)
    assert first.status_code == shuffled.status_code == 200
    assert first.json()['hourly_plan'] == shuffled.json()['hourly_plan']
    assert first.json()['total_cost_bdt'] == shuffled.json()['total_cost_bdt']


BAD_BATCHES = [
    'null', '[]', '{}', '{broken', model_output([]),
    model_output([noop(1)]), model_output([noop(), noop()]),
    model_output([{**noop(), 'applies': 'false'}]),
    model_output([{**noop(), 'note_index': 0.5}]),
    model_output([{**noop(), 'note_index': True}]),
    model_output([{**active(), 'directive_type': 'no_discharge_window'}]),
    model_output([active('solar_reduction', factor=1.5)]),
    model_output([active('minimum_battery_reserve', minimum_energy_kwh=100000)]),
    model_output([active('max_grid_window', max_grid_kwh=float('inf'))]),
    model_output([active('no_charge_window', unexpected=1)]),
    model_output([{**active(), 'structured_adjustment': []}]),
    model_output([{**active(), 'structured_adjustment': {'kind': 'no_charge_window', 'hours': [3, 2]}}]),
    model_output([{**active(), 'structured_adjustment': {'kind': 'no_charge_window', 'hours': [2, 2]}}]),
    model_output([{**active(), 'structured_adjustment': {'kind': 'no_charge_window', 'hours': [True]}}]),
    model_output([{**active(), 'structured_adjustment': {'kind': 'no_charge_window', 'hours': [24]}}]),
]


@pytest.mark.parametrize('raw', BAD_BATCHES)
def test_bad_model_output_is_not_silently_ignored(client, monkeypatch, payload, raw):
    payload['operator_notes'] = ['Do not charge the battery from 2 AM until 4 AM.']
    call = Mock(return_value=raw)
    monkeypatch.setattr(llm, '_call_ollama', call)
    response = client.post('/optimize-energy', json=payload)
    assert response.status_code == 500
    assert response.json()['error'] == 'schedule_unavailable'
    assert 'hourly_plan' not in response.json()
    assert call.call_count == 2


def test_model_retry_repairs_output(monkeypatch, payload):
    call = Mock(side_effect=['[]', model_output([active()])])
    monkeypatch.setattr(llm, '_call_ollama', call)
    result = llm.interpret_notes(['No charge 2 AM to 4 AM.'], Battery(**payload['battery']))
    assert result[0].structured_adjustment.hours == [2, 3]
    assert call.call_count == 2


def test_provider_timeout_is_controlled(client, monkeypatch, payload):
    call = Mock(side_effect=requests.Timeout('sensitive provider detail'))
    monkeypatch.setattr(llm, '_call_ollama', call)
    response = client.post('/optimize-energy', json=payload)
    assert response.status_code == 500
    assert 'sensitive' not in response.text
    assert call.call_count == 2


@pytest.mark.parametrize('model,url,expects_schema', [
    ('gemma4:31b-cloud', 'http://localhost:11434', False),
    ('gemma4:cloud', 'http://localhost:11434', False),
    ('gemma4:31b', 'https://ollama.com', False),
    ('example:small', 'http://localhost:11434', True),
])
def test_cloud_and_local_format_modes(monkeypatch, model, url, expects_schema):
    monkeypatch.setattr(llm, 'OLLAMA_MODEL', model)
    monkeypatch.setattr(llm, 'OLLAMA_URL', url)
    monkeypatch.setenv('OLLAMA_FORMAT', 'auto')
    post = Mock(return_value=Mock(json=lambda: {'message': {'content': model_output([noop()])}}))
    monkeypatch.setattr(llm.requests, 'post', post)
    llm._call_ollama([])
    sent = post.call_args.kwargs['json']
    assert ('format' in sent) == expects_schema
    assert post.call_args.kwargs['timeout'] == (2, llm.REQUEST_TIMEOUT_S)


def test_fractional_directive_is_preserved(client, monkeypatch, payload):
    payload['operator_notes'] = ['Only one third of solar remains 2 AM to 4 AM.']
    payload['hours'][2]['solar_kwh'] = 30
    factor = 0.333333333
    monkeypatch.setattr(llm, '_call_ollama', lambda _: model_output([active('solar_reduction', factor=factor)]))
    response = client.post('/optimize-energy', json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    adjustment = body['directive_interpretation'][0]['structured_adjustment']
    assert adjustment['factor'] == factor
    assert 'kind' not in adjustment
    assert body['hourly_plan'][2]['solar_used_kwh'] <= 30 * factor + 1e-6


def test_zero_rates_are_valid(client, no_directives, payload):
    payload['battery']['max_charge_kwh_per_hour'] = 0
    payload['battery']['max_discharge_kwh_per_hour'] = 0
    response = client.post('/optimize-energy', json=payload)
    assert response.status_code == 200, response.text
    assert all(row['battery_action'] == 'idle' for row in response.json()['hourly_plan'])


def test_infeasible_directive_returns_422(client, monkeypatch, payload):
    payload['operator_notes'] = ['Import no grid energy all day.']
    entry = active('max_grid_window', max_grid_kwh=0)
    entry['structured_adjustment']['hours'] = list(range(24))
    monkeypatch.setattr(llm, '_call_ollama', lambda _: model_output([entry]))
    response = client.post('/optimize-energy', json=payload)
    assert response.status_code == 422
    assert response.json()['error'] == 'infeasible_schedule'


def test_time_limited_incumbent_not_claimed_optimal(monkeypatch, payload):
    import pulp
    def interrupted(problem, solver):
        problem.status = pulp.LpStatusOptimal
        problem.sol_status = pulp.LpSolutionIntegerFeasible
    monkeypatch.setattr(pulp.LpProblem, 'solve', interrupted)
    req = OptimizeRequest(**payload)
    with pytest.raises(SolverFailure):
        optimize(req.hours, req.battery, [])
