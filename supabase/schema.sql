-- Estrutura para armazenar execucoes do modelo e previsoes do case Datarisk.
-- O acesso publico permanece bloqueado: somente uma chave secreta usada pelo
-- script local pode gravar ou consultar estes dados.

create table if not exists public.prediction_runs (
    id uuid primary key,
    model_name text not null,
    validation_auc double precision not null check (validation_auc between 0 and 1),
    validation_brier double precision not null check (validation_brier between 0 and 1),
    prediction_count integer not null check (prediction_count >= 0),
    created_at timestamptz not null default now()
);

create table if not exists public.credit_risk_predictions (
    run_id uuid not null references public.prediction_runs(id) on delete cascade,
    prediction_order integer not null check (prediction_order >= 0),
    client_id text not null,
    vintage text not null,
    default_probability double precision not null check (default_probability between 0 and 1),
    created_at timestamptz not null default now(),
    primary key (run_id, prediction_order)
);

alter table public.prediction_runs enable row level security;
alter table public.credit_risk_predictions enable row level security;

revoke all on table public.prediction_runs from anon, authenticated;
revoke all on table public.credit_risk_predictions from anon, authenticated;
grant all on table public.prediction_runs to service_role;
grant all on table public.credit_risk_predictions to service_role;
