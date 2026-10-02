mkdir -p /app/Acme-Pricelist/lib/Acme /app/Acme-Pricelist/bin /app/Acme-Pricelist/t /data
cd /app/Acme-Pricelist
cat > Makefile.PL <<'EOF'
use strict;
use warnings;
use ExtUtils::MakeMaker;

WriteMakefile(
    NAME         => 'Acme::Pricelist',
    AUTHOR       => 'Acme Tools <tools@acme.example>',
    VERSION_FROM => 'lib/Acme/Pricelist.pm',
    ABSTRACT     => 'price list totals and a report CLI',
    EXE_FILES    => ['bin/pricelist_report'],
    PREREQ_PM    => { 'Exporter' => 0 },
    INSTALLDIRS  => 'site',
);
EOF
cat > lib/Acme/Pricelist.pm <<'EOF'
package Acme::Pricelist;
use strict;
use warnings;
use Exporter 'import';

our $VERSION   = '0.03';
our @EXPORT_OK = qw(parse_lines summarize format_report);

# "sku,qty,unit_price" lines -> list of {sku, qty, price, total}; skips the header, blanks and comments.
sub parse_lines {
    my @items;
    for my $line (@_) {
        chomp(my $l = $line);
        next if $l =~ /^\s*(?:#|$)/ or $l =~ /^sku,/;
        my ($sku, $qty, $price) = split /,/, $l;
        push @items, { sku => $sku, qty => $qty, price => $price, total => $qty * $price };
    }
    return @items;
}

# Items by line total, largest first (ties by sku), with the grand total and the mean line total.
sub summarize {
    my @items = sort { $b->{total} cmp $a->{total} or $a->{sku} cmp $b->{sku} } @_;
    my $sum = 0;
    $sum += $_->{total} for @items;
    return { items => \@items, total => $sum, avg => @items ? $sum / @items : 0 };
}

sub format_report {
    my ($s) = @_;
    my @out = map { sprintf("%-8s %4d x %8.2f = %9.2f", @{$_}{qw(sku qty price total)}) } @{ $s->{items} };
    push @out, sprintf("TOTAL %.2f", $s->{total});
    push @out, sprintf("AVG %d", $s->{avg});
    return join("\n", @out) . "\n";
}

1;
EOF
cat > bin/pricelist-report <<'EOF'
#!/usr/bin/perl
use strict;
use warnings;
use Acme::Pricelist qw(parse_lines summarize format_report);

die "usage: pricelist-report FILE\n" unless @ARGV == 1;
open my $fh, '<', $ARGV[0] or die "pricelist-report: $ARGV[0]: $!\n";
print format_report(summarize(parse_lines(<$fh>)));
EOF
chmod +x bin/pricelist-report
cat > t/01-parse.t <<'EOF'
use strict;
use warnings;
use Test::More tests => 4;
use Acme::Pricelist qw(parse_lines);

my @items = parse_lines("sku,qty,unit_price\n", "# comment\n", "A1,3,4.50\n", "\n", "B2,2,0.25\n");
is(scalar @items, 2, 'header, comment and blank line skipped');
is($items[0]{sku}, 'A1', 'first sku');
is($items[0]{total}, 13.5, 'line total is qty x price');
is($items[1]{total}, 0.5, 'second line total');
EOF
cat > t/02-summary.t <<'EOF'
use strict;
use warnings;
use Test::More tests => 3;
use Acme::Pricelist qw(parse_lines summarize);

my $s = summarize(parse_lines("X,1,9.5\n", "Y,1,100\n", "Z,2,5\n", "W,1,10\n"));
is_deeply([map { $_->{sku} } @{ $s->{items} }], [qw(Y W Z X)], 'largest line total first, ties by sku');
is($s->{total}, 129.5, 'grand total');
is($s->{avg}, 32.375, 'mean line total');
EOF
cat > README <<'EOF'
Acme::Pricelist 0.03

  pricelist-report FILE

FILE has "sku,qty,unit_price" lines (a header line, blank lines and # comments are skipped). The report has one
line per item, largest line total first (equal totals by sku):

  SKU      QTY x    PRICE =     TOTAL          ("%-8s %4d x %8.2f = %9.2f")

then "TOTAL <grand total>" and "AVG <mean line total>", both with two decimals.

Build, test and install (site install: the module under /usr/local/share/perl, the script in /usr/local/bin):

  perl Makefile.PL && make && make test && make install
EOF
cat > /data/prices.csv <<'EOF'
sku,qty,unit_price
A100,3,4.50
B200,10,1.25
C300,1,99.99
D400,12,0.80
E500,2,5.00
EOF
