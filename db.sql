create table if not exists public.products (
  product_id text primary key,
  name text not null,
  category text,
  price_thb numeric,
  stock int,
  description text,
  tags text,
  updated_at timestamptz not null default now()
);

